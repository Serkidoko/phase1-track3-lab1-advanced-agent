from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
from src.reflexion_lab.utils import load_dataset


def _example(qid: str):
    return next(item for item in load_dataset("data/hotpot_mini.json") if item.qid == qid)


def test_reflexion_uses_reflection_memory(monkeypatch):
    monkeypatch.setenv("LLM_MODE", "mock")
    record = ReflexionAgent(max_attempts=3).run(_example("hp2"))

    assert record.is_correct is True
    assert record.attempts == 2
    assert len(record.reflections) == 1
    assert record.traces[0].reflection == record.reflections[0]
    assert record.token_estimate > 0


def test_react_does_not_reflect(monkeypatch):
    monkeypatch.setenv("LLM_MODE", "mock")
    record = ReActAgent().run(_example("hp2"))

    assert record.is_correct is False
    assert record.attempts == 1
    assert record.reflections == []
    assert record.failure_mode == "incomplete_multi_hop"
