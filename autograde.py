from __future__ import annotations
import json
from pathlib import Path
import typer
from rich import print
app = typer.Typer(add_completion=False)
REQUIRED_KEYS = ["meta", "summary", "failure_modes", "examples", "extensions", "discussion"]

@app.command()
def main(report_path: str = "outputs/sample_run/report.json") -> None:
    path = Path(report_path)
    if not path.exists():
        raise typer.BadParameter(f"Missing report file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    
    # Core flow points: 80 total
    # 1. Schema completeness (30 points)
    present = sum(1 for key in REQUIRED_KEYS if key in payload)
    schema_points = round(30 * present / len(REQUIRED_KEYS))
    
    # 2. Experiment completeness (30 points)
    exp_points = 0
    summary = payload.get("summary", {})
    if "react" in summary and "reflexion" in summary:
        exp_points += 10
    if payload.get("meta", {}).get("num_records", 0) >= 100:  # Changed from 16 to 100 as per new requirement
        exp_points += 10
    if len(payload.get("examples", [])) >= 20:
        exp_points += 10
    
    # 3. Analysis depth (20 points)
    analysis_points = 0
    unique_failure_modes = set()
    for value in payload.get("failure_modes", {}).values():
        if isinstance(value, dict):
            unique_failure_modes.update(value.keys())
    if len(unique_failure_modes) >= 3:
        analysis_points += 8
    if len(payload.get("discussion", "")) >= 250:
        analysis_points += 12
    
    flow_score = schema_points + exp_points + analysis_points
    
    # Bonus points: 20 total
    recognized = {"structured_evaluator", "reflection_memory", "benchmark_report_json", "mock_mode_for_autograding", "adaptive_max_attempts", "memory_compression", "mini_lats_branching", "plan_then_execute"}
    bonus_points = min(20, 10 * len(set(payload.get("extensions", [])) & recognized))
    
    total_score = flow_score + bonus_points
    
    print(f"Auto-grade total: {total_score}/100")
    print(f"- Flow Score (Core): {flow_score}/80")
    print(f"  * Schema: {schema_points}/30")
    print(f"  * Experiment: {exp_points}/30")
    print(f"  * Analysis: {analysis_points}/20")
    print(f"- Bonus Score: {bonus_points}/20")
    print("\nManual review required for code quality, actual token logic, and reasoning depth.")

if __name__ == "__main__":
    app()
