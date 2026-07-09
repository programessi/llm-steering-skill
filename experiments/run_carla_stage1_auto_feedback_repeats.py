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

from envs.carla_adapter import CarlaAdapter, CarlaAdapterConfig
from experiments.run_carla_stage1_conditional_policy import (
    ConditionalPolicyProgramGenerator,
    LLMConditionalPolicyProgramGenerator,
    STAGE1_TASKS,
    run_conditional_program,
)
from robot_steering.backends import robot_steering_backend_name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--policy-generator", choices=["deterministic", "llm"], default="llm")
    parser.add_argument("--task", choices=sorted(STAGE1_TASKS), default="right_angle_left")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default=None)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--lane-departure-limit-m", type=float, default=5.0)
    parser.add_argument("--start-distance-to-maneuver-m", type=float, default=8.0)
    parser.add_argument("--max-policy-iterations", type=int, default=80)
    parser.add_argument("--steering-skill", choices=["oracle", "simulated_robot"], default="oracle")
    parser.add_argument("--robot-skill-backend", choices=["stub", "kinematic_valve", "maniskill_valve"], default="stub")
    parser.add_argument("--perception-source", choices=["oracle", "model_marker", "trilitenet_lane_yolo_marker"], default="oracle")
    parser.add_argument("--yolo-model", default=str(ROOT / "models" / "yolo11n.pt"))
    parser.add_argument("--visual-marker-class", default="car")
    parser.add_argument("--visual-marker-real-height-m", type=float, default=1.5)
    parser.add_argument("--visual-marker-confidence", type=float, default=0.25)
    parser.add_argument("--visual-marker-distance-m", type=float, default=8.0)
    parser.add_argument("--visual-marker-lateral-offset-m", type=float, default=2.2)
    parser.add_argument("--visual-marker-blueprint", default="vehicle.mini.cooper_s")
    parser.add_argument("--trilitenet-root", default=str(ROOT / "third_party" / "trilitenet"))
    parser.add_argument("--trilitenet-config", default="small")
    parser.add_argument("--trilitenet-weights", default=str(ROOT / "models" / "trilitenet" / "small.pth"))
    parser.add_argument("--trilitenet-device", default="cpu")
    parser.add_argument("--trilitenet-input-size", type=int, default=640)
    args = parser.parse_args()

    out_root = Path(args.out) if args.out else ROOT / "runs" / "auto_feedback_tasks" / args.task / "repeats"
    out_root.mkdir(parents=True, exist_ok=True)
    task = STAGE1_TASKS[args.task]
    generator = (
        LLMConditionalPolicyProgramGenerator(generator_name="openai_compatible_llm_generator")
        if args.policy_generator == "llm"
        else ConditionalPolicyProgramGenerator(generator_name="deterministic_stage1_conditional_generator")
    )
    config = CarlaAdapterConfig(
        host=args.host,
        port=args.port,
        town=args.town,
        spawn_index=task.spawn_index,
        width=args.width,
        height=args.height,
        fixed_delta_seconds=1.0 / float(args.fps),
        timeout_s=args.timeout_s,
        lane_departure_limit_m=args.lane_departure_limit_m,
    )

    rows: list[dict] = []
    env = CarlaAdapter(config)
    try:
        for trial_idx in range(1, args.trials + 1):
            trial_dir = out_root / f"trial_{trial_idx:03d}"
            attempt1_program = generator.generate_conditional(
                task,
                "conditional_policy_feedback_retry",
                feedback_event="none",
                attempt=1,
            )
            attempt1 = run_conditional_program(
                env,
                task,
                attempt1_program,
                args,
                trial_dir / "attempt_01",
            )

            feedback_event = str(attempt1["execution_feedback_event"])
            attempt2 = None
            final = attempt1
            if feedback_event != "none":
                attempt2_program = generator.generate_conditional(
                    task,
                    "conditional_policy_feedback_retry",
                    feedback_event=feedback_event,
                    attempt=2,
                )
                attempt2 = run_conditional_program(
                    env,
                    task,
                    attempt2_program,
                    args,
                    trial_dir / "attempt_02",
                )
                final = attempt2

            trial_summary = {
                "trial": trial_idx,
                "attempt_01": attempt1,
                "attempt_02": attempt2,
                "final": final,
                "feedback_corrections": int(feedback_event != "none"),
                "retry_attempts": int(attempt2 is not None),
            }
            (trial_dir / "summary.json").write_text(json.dumps(trial_summary, indent=2))
            rows.append(_trial_row(trial_idx, attempt1, attempt2, final))
    finally:
        env.close()

    write_csv(out_root / "trials.csv", rows)
    aggregate = aggregate_rows(rows, args)
    (out_root / "summary.json").write_text(json.dumps(aggregate, indent=2))
    print(json.dumps(aggregate, indent=2))


def _trial_row(trial_idx: int, attempt1: dict, attempt2: dict | None, final: dict) -> dict:
    return {
        "trial": trial_idx,
        "attempt1_success": int(bool(attempt1["success"])),
        "attempt1_feedback": attempt1["execution_feedback_event"],
        "attempt1_valid_code": int(bool(attempt1["policy_runtime_valid_code"])),
        "retry_attempted": int(attempt2 is not None),
        "final_success": int(bool(final["success"])),
        "final_feedback": final["execution_feedback_event"],
        "final_attempt": int(final["generation_attempt"]),
        "final_valid_code": int(bool(final["policy_runtime_valid_code"])),
        "final_primitive_call_count": int(final["primitive_call_count"]),
        "final_mean_lane_center_offset_m": f"{float(final['mean_lane_center_offset_m']):.4f}",
        "final_max_lane_center_offset_m": f"{float(final['max_lane_center_offset_m']):.4f}",
        "final_policy_generator": final["policy_generator"],
    }


def aggregate_rows(rows: list[dict], args: argparse.Namespace) -> dict:
    count = len(rows)
    feedback_counts = Counter(str(row["attempt1_feedback"]) for row in rows)
    final_feedback_counts = Counter(str(row["final_feedback"]) for row in rows)
    return {
        "trials": count,
        "task": args.task,
        "policy_generator": args.policy_generator,
        "steering_skill": args.steering_skill,
        "steering_skill_impl": steering_skill_impl_name(args),
        "robot_skill_backend": args.robot_skill_backend,
        "robot_skill_impl": robot_steering_backend_name(args.robot_skill_backend),
        "perception_source": args.perception_source,
        "visual_marker_class": args.visual_marker_class,
        "attempt1_success_rate": _rate(rows, "attempt1_success"),
        "retry_rate": _rate(rows, "retry_attempted"),
        "final_success_rate": _rate(rows, "final_success"),
        "valid_code_rate": _rate(rows, "final_valid_code"),
        "attempt1_feedback_counts": dict(feedback_counts),
        "final_feedback_counts": dict(final_feedback_counts),
        "mean_final_max_lane_center_offset_m": _mean_float(rows, "final_max_lane_center_offset_m"),
        "trials_csv": str((Path(args.out) if args.out else ROOT / "runs" / "auto_feedback_tasks" / args.task / "repeats") / "trials.csv"),
    }


def steering_skill_impl_name(args: argparse.Namespace) -> str:
    if getattr(args, "steering_skill", "oracle") == "simulated_robot":
        return "SimulatedRobotSteeringSkill"
    return "OracleSteeringSkill"


def _rate(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0
    return float(sum(int(row[key]) for row in rows) / len(rows))


def _mean_float(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0
    return float(sum(float(row[key]) for row in rows) / len(rows))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
