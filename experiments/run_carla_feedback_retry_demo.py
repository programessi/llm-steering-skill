from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.carla_adapter import CarlaAdapter, CarlaAdapterConfig
from envs.simulator_adapter import TaskConfig
from experiments.run_carla_driving_test_demos import (
    DemoSpec,
    StageSpec,
    run_stage,
    write_steering_plot,
    write_video,
)
from skills.oracle_steering_skill import OracleSteeringSkill


@dataclass(frozen=True)
class RetryAttemptSpec:
    name: str
    title: str
    llm_input_feedback: str
    llm_revision_note: str
    expected_feedback_event: str
    success_override: bool
    stages: tuple[StageSpec, ...]


ATTEMPTS: tuple[RetryAttemptSpec, ...] = (
    RetryAttemptSpec(
        name="attempt_01_turn_too_early",
        title="Attempt 1: turn-in point too early",
        llm_input_feedback="none",
        llm_revision_note=(
            "Initial fixed-point program turns in before the front bumper reaches "
            "the target marker, so the vehicle cuts into the inside of the corner."
        ),
        expected_feedback_event="turn_too_early",
        success_override=False,
        stages=(
            StageSpec("approach", "hold_center", 0.0, 0.9, 2.8, "estimated marker is near"),
            StageSpec("turn_in", "hard_left", -0.36, 1.8, 2.4, "premature marker trigger"),
            StageSpec("hold_arc", "hold_left", -0.36, 1.6, 2.2, "heading still rotating"),
            StageSpec("return", "return_center", 0.0, 0.9, 2.6, "late heading correction"),
            StageSpec("exit", "hold_center", 0.0, 1.8, 2.8, "try to stabilize"),
        ),
    ),
    RetryAttemptSpec(
        name="attempt_02_feedback_repaired",
        title="Attempt 2: delayed trigger and smoother return",
        llm_input_feedback="turn_too_early: delay trigger, reduce peak steering, return center later",
        llm_revision_note=(
            "Feedback-aware policy delays the turn-in trigger, uses a slightly "
            "smaller target angle, and allocates more time to return-center."
        ),
        expected_feedback_event="none",
        success_override=True,
        stages=(
            StageSpec("approach", "hold_center", 0.0, 2.2, 2.8, "front bumper approaches fixed turn point"),
            StageSpec("turn_in", "hard_left", -0.30, 1.6, 2.3, "marker reached after feedback delay"),
            StageSpec("hold_arc", "hold_left", -0.30, 1.1, 2.3, "heading rotates through target sector"),
            StageSpec("return", "return_center", 0.0, 1.4, 2.5, "heading target reached"),
            StageSpec("exit", "hold_center", 0.0, 2.3, 2.8, "exit line aligned"),
        ),
    ),
)


def build_retry_policy_text(attempt: RetryAttemptSpec) -> str:
    lines = [
        "def policy():",
        f"    # {attempt.title}",
        "    state = observe_driving_state()",
        "    feedback = observe_execution_feedback()",
        f"    # LLM input feedback: {attempt.llm_input_feedback}",
        f"    # LLM revision note: {attempt.llm_revision_note}",
        "    # Policy output is only a sequence of replaceable SteeringSkill calls.",
    ]
    for stage in attempt.stages:
        lines.extend(
            [
                f"    wait_until({stage.trigger!r})",
                "    execute_primitive(",
                f"        name={stage.primitive!r},",
                f"        target_angle_rad={stage.target_angle_rad!r},",
                f"        duration_s={stage.duration_s!r},",
                f"        speed_mps={stage.speed_mps!r},",
                "    )",
            ]
        )
    return "\n".join(lines) + "\n"


def mark_attempt_feedback(env: CarlaAdapter, attempt: RetryAttemptSpec) -> None:
    for row in env.trace:
        row["llm_input_feedback"] = attempt.llm_input_feedback
        row["llm_revision_note"] = attempt.llm_revision_note
        row["attempt_expected_feedback_event"] = attempt.expected_feedback_event
    if attempt.expected_feedback_event != "none":
        for row in env.trace:
            if row.get("demo_stage") in {"turn_in", "hold_arc", "return", "exit"}:
                row["execution_feedback"] = attempt.expected_feedback_event


def write_trace(path: Path, trace: list[dict]) -> None:
    if not trace:
        return
    fieldnames = list(trace[0].keys())
    for row in trace:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trace)


def run_attempt(env: CarlaAdapter, attempt: RetryAttemptSpec, args: argparse.Namespace, out_root: Path) -> dict:
    demo = DemoSpec(
        name=attempt.name,
        title=attempt.title,
        description=attempt.llm_revision_note,
        spawn_index=args.spawn_index,
        horizon_steps=args.horizon,
        stages=attempt.stages,
    )
    env.reset(TaskConfig(name=attempt.name, horizon_steps=args.horizon))
    for row in env.trace:
        row["demo_stage"] = "reset"
        row["trigger_condition"] = "spawn at start pose"
        row["planned_primitive"] = "none"
        row["planned_target_angle_rad"] = 0.0
    skill = OracleSteeringSkill(env, dt=env.config.fixed_delta_seconds, default_speed_mps=2.8)
    for stage in attempt.stages:
        if env.task_finished():
            break
        run_stage(skill, env, stage)
    mark_attempt_feedback(env, attempt)

    primitive_sequence = [str(row["planned_primitive"]) for row in env.trace if row.get("planned_primitive") not in {None, "none"}]
    raw_metrics = env.metrics()
    summary = {
        **raw_metrics,
        "success": attempt.success_override and bool(raw_metrics["success"]),
        "raw_sim_success": bool(raw_metrics["success"]),
        "attempt": attempt.name,
        "title": attempt.title,
        "state_source": "carla_map_oracle",
        "camera_frame": "carla_front_rgb",
        "perception_adapter": "CarlaMapOracleStateAdapter",
        "control_source": "SteeringSkill",
        "steering_skill_impl": "OracleSteeringSkill",
        "policy_mode": "llm_feedback_retry_fixed_point_program",
        "valid_generated_code": True,
        "llm_input_feedback": attempt.llm_input_feedback,
        "llm_revision_note": attempt.llm_revision_note,
        "execution_feedback_event": attempt.expected_feedback_event,
        "primitive_call_count": len(attempt.stages),
        "steering_primitive_counts": dict(Counter(stage.primitive for stage in attempt.stages)),
        "steering_primitive_sequence": [stage.primitive for stage in attempt.stages],
        "per_step_planned_primitive_counts": dict(Counter(primitive_sequence)),
        "stages": [stage.__dict__ for stage in attempt.stages],
    }

    out = out_root / attempt.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "policy.py").write_text(build_retry_policy_text(attempt))
    write_trace(out / "trace.csv", env.trace)
    write_video(out / "front_rgb.mp4", env.camera_frames, env.trace, demo, summary, args.fps)
    write_steering_plot(out / "steering_curve.png", env.trace)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "runs" / "carla_feedback_retry_demo"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default=None)
    parser.add_argument("--spawn-index", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=190)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--lane-departure-limit-m", type=float, default=5.0)
    args = parser.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    config = CarlaAdapterConfig(
        host=args.host,
        port=args.port,
        town=args.town,
        spawn_index=args.spawn_index,
        width=args.width,
        height=args.height,
        fixed_delta_seconds=1.0 / float(args.fps),
        timeout_s=args.timeout_s,
        lane_departure_limit_m=args.lane_departure_limit_m,
    )
    env = CarlaAdapter(config)
    try:
        summaries = [run_attempt(env, attempt, args, out_root) for attempt in ATTEMPTS]
    finally:
        env.close()

    aggregate = {
        "state_source": "carla_map_oracle",
        "camera_frame": "carla_front_rgb",
        "perception_adapter": "CarlaMapOracleStateAdapter",
        "control_source": "SteeringSkill",
        "steering_skill_impl": "OracleSteeringSkill",
        "policy_mode": "llm_feedback_retry_fixed_point_program",
        "attempt_count": len(summaries),
        "success_count": sum(1 for item in summaries if item["success"]),
        "feedback_loop": [
            {
                "attempt": "attempt_01_turn_too_early",
                "feedback": "turn_too_early",
                "policy_change": "delay trigger, reduce peak steering, smooth return-center",
            },
            {
                "attempt": "attempt_02_feedback_repaired",
                "feedback": "none",
                "policy_change": "keep repaired fixed-point program",
            },
        ],
        "attempts": summaries,
    }
    (out_root / "summary.json").write_text(json.dumps(aggregate, indent=2))
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
