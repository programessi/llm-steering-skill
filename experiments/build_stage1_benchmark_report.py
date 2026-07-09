from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_TASK_RUNS = {
    "right_angle_left": ROOT / "runs" / "carla_stage1_auto_feedback_repeats_llm3",
    "lane_change_left": ROOT / "runs" / "auto_feedback_tasks" / "lane_change_left" / "repeats_llm3",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "runs" / "stage1_benchmark_report"))
    parser.add_argument(
        "--task-run",
        action="append",
        default=[],
        help="Override task input as task_name=/abs/or/relative/path. Can be repeated.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    task_runs = dict(DEFAULT_TASK_RUNS)
    for override in args.task_run:
        if "=" not in override:
            raise ValueError(f"--task-run must be task=path, got {override!r}")
        task, path = override.split("=", 1)
        task_runs[task] = Path(path)

    task_summaries = []
    trial_rows = []
    for task, run_dir in task_runs.items():
        run_dir = run_dir if run_dir.is_absolute() else ROOT / run_dir
        summary = summarize_task(task, run_dir)
        task_summaries.append(summary)
        trial_rows.extend(load_trial_rows(task, run_dir))

    write_csv(out_dir / "summary.csv", task_summaries)
    write_csv(out_dir / "trials.csv", trial_rows)
    report = {
        "task_count": len(task_summaries),
        "tasks": task_summaries,
        "trial_count": len(trial_rows),
        "system_boundary": {
            "camera_frame": "carla_front_rgb for videos",
            "driving_state": "CARLA/map oracle plus fixed-point distance adapter",
            "steering_skill": "OracleSteeringSkill behind replaceable SteeringSkill boundary",
            "policy_generator": "OpenAI-compatible LLM API, not codex-a CLI",
            "action_api": "execute_primitive(...) only",
            "feedback": "trace/metrics -> ExecutionFeedback via feedback/trace_feedback_adapter.py",
        },
        "summary_csv": str(out_dir / "summary.csv"),
        "trials_csv": str(out_dir / "trials.csv"),
        "readme": str(out_dir / "README.md"),
    }
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2))
    (out_dir / "README.md").write_text(build_readme(report, trial_rows))
    print(json.dumps(report, indent=2))


def summarize_task(task: str, run_dir: Path) -> dict:
    rows = load_trial_rows(task, run_dir)
    if not rows:
        raise RuntimeError(f"No trial rows found for {task}: {run_dir}")
    attempt1_feedback = Counter(row["attempt1_feedback"] for row in rows)
    final_feedback = Counter(row["final_feedback"] for row in rows)
    return {
        "task": task,
        "run_dir": str(run_dir),
        "trials": len(rows),
        "attempt1_success_rate": f"{rate(rows, 'attempt1_success'):.3f}",
        "retry_rate": f"{rate(rows, 'retry_attempted'):.3f}",
        "final_success_rate": f"{rate(rows, 'final_success'):.3f}",
        "valid_code_rate": f"{rate(rows, 'final_valid_code'):.3f}",
        "attempt1_feedback_counts": json.dumps(dict(attempt1_feedback), sort_keys=True),
        "final_feedback_counts": json.dumps(dict(final_feedback), sort_keys=True),
        "mean_final_max_lane_center_offset_m": f"{mean_float(rows, 'final_max_lane_center_offset_m'):.4f}",
        "demo_attempt1_video": rel_to_report(run_dir / "trial_001" / "attempt_01" / "front_rgb.mp4"),
        "demo_attempt2_video": rel_to_report(run_dir / "trial_001" / "attempt_02" / "front_rgb.mp4"),
        "demo_attempt2_policy": rel_to_report(run_dir / "trial_001" / "attempt_02" / "policy.py"),
    }


def load_trial_rows(task: str, run_dir: Path) -> list[dict]:
    path = run_dir / "trials.csv"
    with path.open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["task"] = task
        row["run_dir"] = str(run_dir)
    return rows


def build_readme(report: dict, trial_rows: list[dict]) -> str:
    lines = [
        "# Stage 1 Benchmark Report",
        "",
        "This report summarizes the current fixed-point driving demonstrations.",
        "",
        "## System Boundary",
        "",
        "- LLM writes restricted Python policy code.",
        "- Policy code reads `DrivingState` and `ExecutionFeedback`.",
        "- Policy code can only act through `execute_primitive(...)`.",
        "- `execute_primitive(...)` calls `OracleSteeringSkill` in CARLA today.",
        "- Videos are CARLA front RGB.",
        "- `DrivingState` is still CARLA/map oracle plus fixed-point distance, not learned perception.",
        "",
        "## Summary",
        "",
        "| Task | Trials | Attempt 1 Success | Retry Rate | Final Success | Attempt 1 Feedback | Final Feedback | Mean Final Max Offset |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- | ---: |",
    ]
    for task in report["tasks"]:
        lines.append(
            "| {task} | {trials} | {a1} | {retry} | {final} | `{a1fb}` | `{ffb}` | {offset} |".format(
                task=task["task"],
                trials=task["trials"],
                a1=task["attempt1_success_rate"],
                retry=task["retry_rate"],
                final=task["final_success_rate"],
                a1fb=task["attempt1_feedback_counts"],
                ffb=task["final_feedback_counts"],
                offset=task["mean_final_max_lane_center_offset_m"],
            )
        )
    lines.extend(
        [
            "",
            "## Demo Links",
            "",
        ]
    )
    for task in report["tasks"]:
        lines.extend(
            [
                f"### {task['task']}",
                "",
                f"- Attempt 1 video: [{task['demo_attempt1_video']}]({task['demo_attempt1_video']})",
                f"- Attempt 2 video: [{task['demo_attempt2_video']}]({task['demo_attempt2_video']})",
                f"- Attempt 2 policy: [{task['demo_attempt2_policy']}]({task['demo_attempt2_policy']})",
                "",
            ]
        )
    lines.extend(
        [
            "## Per-Trial Data",
            "",
            "See [summary.csv](summary.csv), [trials.csv](trials.csv), and [summary.json](summary.json).",
            "",
        ]
    )
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def rate(rows: list[dict], key: str) -> float:
    return sum(int(row[key]) for row in rows) / len(rows)


def mean_float(rows: list[dict], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def rel_to_report(path: Path) -> str:
    return str(Path("..") / path.relative_to(ROOT / "runs"))


if __name__ == "__main__":
    main()

