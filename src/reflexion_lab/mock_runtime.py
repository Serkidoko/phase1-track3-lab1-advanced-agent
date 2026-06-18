from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM
from .schemas import JudgeResult, QAExample, ReflectionEntry
from .utils import normalize_answer

FIRST_ATTEMPT_WRONG = {"hp2": "London", "hp4": "Atlantic Ocean", "hp6": "Red Sea", "hp8": "Andes"}
FAILURE_MODE_BY_QID = {
    "hp2": "incomplete_multi_hop",
    "hp4": "wrong_final_answer",
    "hp6": "entity_drift",
    "hp8": "entity_drift",
}

_ENV_LOADED = False
_CALL_STATS: list["CallStats"] = []


@dataclass(frozen=True)
class LLMConfig:
    api_key: str | None
    chat_url: str | None
    model: str | None
    temperature: float = 0.0
    timeout_s: float = 60.0


@dataclass(frozen=True)
class CallStats:
    call_type: str
    token_count: int
    latency_ms: int
    mode: str
    model: str


def reset_call_stats() -> None:
    _CALL_STATS.clear()


def consume_call_stats() -> list[CallStats]:
    stats = list(_CALL_STATS)
    _CALL_STATS.clear()
    return stats


def runtime_mode() -> str:
    return "llm" if _should_use_llm() else "mock"


def _env_files() -> list[Path]:
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [Path.cwd() / ".env", repo_root / ".env"]
    unique: list[Path] = []
    for path in candidates:
        if path not in unique:
            unique.append(path)
    return unique


def _load_env_file() -> str:
    global _ENV_LOADED
    raw_parts: list[str] = []
    for path in _env_files():
        if not path.exists():
            continue
        raw = path.read_text(encoding="utf-8", errors="ignore")
        raw_parts.append(raw)
        if _ENV_LOADED:
            continue
        for line in raw.splitlines():
            match = re.match(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$", line)
            if not match:
                continue
            key, value = match.groups()
            os.environ.setdefault(key, value.strip().strip("'\""))
    _ENV_LOADED = True
    return "\n".join(raw_parts)


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def _chat_url_from_base(base_url: str | None) -> str | None:
    if not base_url:
        return None
    base_url = base_url.strip().rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    if base_url.endswith("/v1"):
        return f"{base_url}/chat/completions"
    return f"{base_url}/v1/chat/completions"


def _scan_raw_env(raw_env: str) -> tuple[str | None, str | None, str | None]:
    url_match = re.search(r"https?://[^\s'\"\\]+/v\d+/chat/completions", raw_env)
    token_match = re.search(r"Authorization\s*:\s*Bearer\s+([^\s'\"\\]+)", raw_env, re.IGNORECASE)
    model_match = re.search(r"""["']model["']\s*:\s*["']([^"']+)["']""", raw_env)

    raw_key = None
    for line in raw_env.splitlines():
        candidate = line.strip().strip("'\"")
        if not candidate or "=" in candidate or " " in candidate or candidate.lower().startswith("curl"):
            continue
        if len(candidate) >= 24:
            raw_key = candidate
            break

    api_key = token_match.group(1) if token_match else raw_key
    if api_key and api_key.startswith("$"):
        api_key = os.getenv(api_key[1:])
    return api_key, url_match.group(0) if url_match else None, model_match.group(1) if model_match else None


def _llm_config() -> LLMConfig:
    raw_env = _load_env_file()
    raw_api_key, raw_chat_url, raw_model = _scan_raw_env(raw_env)
    base_url = _first_env("LLM_BASE_URL", "OPENAI_BASE_URL", "OPENROUTER_BASE_URL", "XAI_BASE_URL")
    chat_url = _first_env("LLM_CHAT_URL", "OPENAI_CHAT_URL") or raw_chat_url or _chat_url_from_base(base_url)
    api_key = _first_env(
        "LLM_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "XAI_API_KEY",
        "GROQ_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
    ) or raw_api_key
    model = _first_env("LLM_MODEL", "OPENAI_MODEL", "OPENROUTER_MODEL", "XAI_MODEL") or raw_model or "gpt-4o-mini"
    temperature = float(os.getenv("LLM_TEMPERATURE", "0"))
    timeout_s = float(os.getenv("LLM_TIMEOUT_S", "60"))
    return LLMConfig(api_key=api_key, chat_url=chat_url, model=model, temperature=temperature, timeout_s=timeout_s)


def _env_mode() -> str:
    return os.getenv("LLM_MODE", os.getenv("RUNTIME_MODE", "auto")).strip().lower()


def _should_use_llm() -> bool:
    mode = _env_mode()
    if mode in {"mock", "offline"}:
        return False
    config = _llm_config()
    configured = bool(config.api_key and config.chat_url and config.model)
    if configured:
        return True
    if mode in {"llm", "real"}:
        missing = []
        if not config.api_key:
            missing.append("API key")
        if not config.chat_url:
            missing.append("chat completions URL")
        raise RuntimeError(f"LLM_MODE={mode} but missing {', '.join(missing)}. Set LLM_API_KEY and LLM_BASE_URL/LLM_CHAT_URL.")
    return False


def _estimate_tokens(*parts: str) -> int:
    text = "\n".join(part for part in parts if part)
    return max(1, len(text) // 4)


def _record(call_type: str, token_count: int, latency_ms: int, mode: str, model: str) -> None:
    _CALL_STATS.append(CallStats(call_type=call_type, token_count=token_count, latency_ms=latency_ms, mode=mode, model=model))


def _record_mock(call_type: str, *parts: str) -> None:
    _record(call_type, _estimate_tokens(*parts), 0, "mock", "mock")


def _format_context(example: QAExample) -> str:
    return "\n".join(f"[{i}] {chunk.title}: {chunk.text}" for i, chunk in enumerate(example.context, start=1))


def _redact(text: str, config: LLMConfig) -> str:
    if config.api_key:
        text = text.replace(config.api_key, "<redacted>")
    return text


def _chat_completion(call_type: str, system: str, user: str, *, max_tokens: int) -> str:
    config = _llm_config()
    if not (config.api_key and config.chat_url and config.model):
        raise RuntimeError("LLM is not configured.")

    payload: dict[str, Any] = {
        "model": config.model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": config.temperature,
        "max_tokens": max_tokens,
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        config.chat_url,
        data=body,
        headers={"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"},
        method="POST",
    )

    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=config.timeout_s) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM request failed with HTTP {exc.code}: {_redact(detail, config)}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM request failed: {exc.reason}") from exc

    latency_ms = int((time.perf_counter() - started) * 1000)
    data = json.loads(response_body)
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage") or {}
    token_count = usage.get("total_tokens")
    if not isinstance(token_count, int):
        prompt_tokens = usage.get("prompt_tokens", 0) if isinstance(usage.get("prompt_tokens", 0), int) else 0
        completion_tokens = usage.get("completion_tokens", 0) if isinstance(usage.get("completion_tokens", 0), int) else 0
        token_count = prompt_tokens + completion_tokens or _estimate_tokens(system, user, content)
    _record(call_type, token_count, latency_ms, "llm", config.model)
    return content


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object.")
    return parsed


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _clean_answer(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|text)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        parsed = _extract_json(text)
        for key in ("answer", "final_answer", "predicted_answer"):
            if key in parsed:
                return str(parsed[key]).strip()
    except Exception:
        pass
    text = re.sub(r"^(final answer|answer)\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    return " ".join(line.strip() for line in text.splitlines() if line.strip())


def _mock_actor_answer(example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> str:
    if example.qid not in FIRST_ATTEMPT_WRONG:
        answer = example.gold_answer
    elif agent_type == "react":
        answer = FIRST_ATTEMPT_WRONG[example.qid]
    elif attempt_id == 1 and not reflection_memory:
        answer = FIRST_ATTEMPT_WRONG[example.qid]
    else:
        answer = example.gold_answer
    _record_mock("actor", example.question, _format_context(example), "\n".join(reflection_memory), answer)
    return answer


def _mock_evaluator(example: QAExample, answer: str) -> JudgeResult:
    if normalize_answer(example.gold_answer) == normalize_answer(answer):
        judge = JudgeResult(score=1, reason="Final answer matches the gold answer after normalization.")
    elif normalize_answer(answer) == "london":
        judge = JudgeResult(
            score=0,
            reason="The answer stopped at the birthplace city and never completed the second hop to the river.",
            missing_evidence=["Need to identify the river that flows through London."],
            spurious_claims=[],
        )
    else:
        judge = JudgeResult(
            score=0,
            reason="The final answer selected the wrong second-hop entity.",
            missing_evidence=["Need to ground the answer in the second paragraph."],
            spurious_claims=[answer],
        )
    _record_mock("evaluator", example.question, example.gold_answer, answer, judge.reason)
    return judge


def _mock_reflector(example: QAExample, attempt_id: int, judge: JudgeResult) -> ReflectionEntry:
    strategy = "Do the second hop explicitly: birthplace city -> river through that city." if example.qid == "hp2" else "Verify the final entity against the second paragraph before answering."
    reflection = ReflectionEntry(
        attempt_id=attempt_id,
        failure_reason=judge.reason,
        lesson="A partial first-hop answer is not enough; the final answer must complete all hops.",
        next_strategy=strategy,
    )
    _record_mock("reflector", example.question, judge.reason, reflection.lesson, reflection.next_strategy)
    return reflection


def actor_answer(example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> str:
    if not _should_use_llm():
        return _mock_actor_answer(example, attempt_id, agent_type, reflection_memory)

    memory = "\n".join(f"- {item}" for item in reflection_memory) or "None"
    user = f"""Question:
{example.question}

Context:
{_format_context(example)}

Reflection memory:
{memory}

Attempt: {attempt_id}
Agent type: {agent_type}
"""
    return _clean_answer(_chat_completion("actor", ACTOR_SYSTEM, user, max_tokens=128))


def evaluator(example: QAExample, answer: str) -> JudgeResult:
    if not _should_use_llm():
        return _mock_evaluator(example, answer)

    user = f"""Question:
{example.question}

Context:
{_format_context(example)}

Gold answer:
{example.gold_answer}

Predicted answer:
{answer}
"""
    raw = _chat_completion("evaluator", EVALUATOR_SYSTEM, user, max_tokens=384)
    try:
        data = _extract_json(raw)
        return JudgeResult(
            score=1 if int(data.get("score", 0)) == 1 else 0,
            reason=str(data.get("reason") or "No reason provided."),
            missing_evidence=_as_list(data.get("missing_evidence")),
            spurious_claims=_as_list(data.get("spurious_claims")),
        )
    except Exception as exc:
        return JudgeResult(score=0, reason=f"Evaluator returned malformed JSON: {exc}", missing_evidence=[], spurious_claims=[answer])


def reflector(example: QAExample, attempt_id: int, judge: JudgeResult) -> ReflectionEntry:
    if not _should_use_llm():
        return _mock_reflector(example, attempt_id, judge)

    user = f"""Question:
{example.question}

Context:
{_format_context(example)}

Failed attempt id:
{attempt_id}

Evaluator reason:
{judge.reason}

Missing evidence:
{json.dumps(judge.missing_evidence)}

Spurious claims:
{json.dumps(judge.spurious_claims)}
"""
    raw = _chat_completion("reflector", REFLECTOR_SYSTEM, user, max_tokens=384)
    try:
        data = _extract_json(raw)
        return ReflectionEntry(
            attempt_id=int(data.get("attempt_id", attempt_id)),
            failure_reason=str(data.get("failure_reason") or judge.reason),
            lesson=str(data.get("lesson") or "Use all relevant context before answering."),
            next_strategy=str(data.get("next_strategy") or "Trace each hop from the question to the final answer."),
        )
    except Exception as exc:
        return ReflectionEntry(
            attempt_id=attempt_id,
            failure_reason=judge.reason,
            lesson=f"Reflector returned malformed JSON: {exc}",
            next_strategy="Re-read the context and answer only after completing every hop.",
        )
