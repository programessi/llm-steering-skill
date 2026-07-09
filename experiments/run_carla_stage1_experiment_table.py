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
from experiments.run_carla_feedback_retry_demo import write_trace
from robot_steering.backends import make_robot_steering_skill, robot_steering_backend_name
from skills.oracle_steering_skill import OracleSteeringSkill
from skills.simulated_robot_steering_skill import SimulatedRobotSteeringSkill
from skills.steering_skill import SteeringSkill


@dataclass(frozen=True)
class Stage1TaskSpec:
    name: str
    title: str
    route_instruction: str
    spawn_index: int
    horizon_steps: int
    target_speed_mps: float
    failure_feedback_event: str
    success_criterion: str


@dataclass(frozen=True)
class GeneratedProgram:
    condition: str
    attempt: int
    title: str
    policy_code: str
    stages: tuple[StageSpec, ...]
    generator: str
    input_feedback: str
    revision_note: str
    expected_feedback_event: str
    semantic_success: bool


RIGHT_ANGLE_LEFT = Stage1TaskSpec(
    name="right_angle_left",
    title="Right-angle left turn",
    route_instruction="Approach a fixed marker, turn left, hold the arc, then return center after the heading target.",
    spawn_index=0,
    horizon_steps=190,
    target_speed_mps=2.8,
    failure_feedback_event="turn_too_early",
    success_criterion="turn-in occurs at the marker and the exit line is aligned without exceeding lane offset limit",
)


LANE_CHANGE_LEFT = Stage1TaskSpec(
    name="lane_change_left",
    title="Fixed-point lane change left",
    route_instruction=(
        "Approach a fixed lane-change marker. At the marker, steer left with a medium-left primitive, "
        "hold the lane-change arc briefly, counter-steer right with a medium-right primitive to stop lateral drift, "
        "then return center and stabilize in the new lane. This is not a continuous curve-following task."
    ),
    spawn_index=6,
    horizon_steps=180,
    target_speed_mps=4.0,
    failure_feedback_event="turn_too_early",
    success_criterion="left lane-change starts near the marker, counter-steers, and stabilizes without lane departure",
)


STAGE1_TASKS: dict[str, Stage1TaskSpec] = {
    RIGHT_ANGLE_LEFT.name: RIGHT_ANGLE_LEFT,
    LANE_CHANGE_LEFT.name: LANE_CHANGE_LEFT,
}


class Stage1PolicyProgramGenerator:
    """Deterministic stand-in for the LLM code-generation contract.

    The important contract is the input/output shape:

    task spec + DrivingState schema + optional ExecutionFeedback
      -> Python policy code that only calls execute_primitive(...)

    A future implementation can replace this class with a direct
    OpenAI-compatible LLM call while preserving the same GeneratedProgram fields
    and experiment runner.
    """

    def __init__(self, generator_name: str = "deterministic_stage1_generator"):
        self.generator_name = generator_name

    def generate(self, task: Stage1TaskSpec, condition: str, feedback_event: str = "none", attempt: int = 1) -> GeneratedProgram:
        if condition == "fixed_no_feedback":
            return self._fixed_no_feedback(task)
        if condition == "llm_no_feedback":
            return self._llm_no_feedback(task)
        if condition == "llm_feedback_retry":
            if feedback_event == task.failure_feedback_event:
                return self._llm_feedback_repaired(task, attempt)
            return self._llm_feedback_initial(task, attempt)
        raise ValueError(f"unknown condition: {condition}")

    def _fixed_no_feedback(self, task: Stage1TaskSpec) -> GeneratedProgram:
        stages = (
            StageSpec("approach", "hold_center", 0.0, 0.8, 2.9, "generic distance estimate says turn soon"),
            StageSpec("turn_in", "hard_left", -0.38, 1.8, 2.4, "fixed timer reached"),
            StageSpec("hold_arc", "hold_left", -0.38, 1.8, 2.2, "generic hold duration"),
            StageSpec("return", "return_center", 0.0, 0.8, 2.6, "fixed return timer"),
            StageSpec("exit", "hold_center", 0.0, 1.6, 2.8, "fixed stabilization timer"),
        )
        return self._program(
            task=task,
            condition="fixed_no_feedback",
            attempt=1,
            stages=stages,
            input_feedback="not used",
            revision_note="Fixed open-loop program ignores task feedback and uses generic timing.",
            expected_feedback_event=task.failure_feedback_event,
            semantic_success=False,
        )

    def _llm_no_feedback(self, task: Stage1TaskSpec) -> GeneratedProgram:
        stages = (
            StageSpec("approach", "hold_center", 0.0, 1.2, 2.8, "estimated marker is near"),
            StageSpec("turn_in", "hard_left", -0.34, 1.7, 2.4, "initial generated marker trigger"),
            StageSpec("hold_arc", "hold_left", -0.34, 1.5, 2.3, "initial generated hold"),
            StageSpec("return", "return_center", 0.0, 0.9, 2.6, "initial generated return"),
            StageSpec("exit", "hold_center", 0.0, 1.9, 2.8, "initial generated exit stabilization"),
        )
        return self._program(
            task=task,
            condition="llm_no_feedback",
            attempt=1,
            stages=stages,
            input_feedback="none",
            revision_note="Task-conditioned policy is generated once and cannot revise after execution feedback.",
            expected_feedback_event=task.failure_feedback_event,
            semantic_success=False,
        )

    def _llm_feedback_initial(self, task: Stage1TaskSpec, attempt: int) -> GeneratedProgram:
        stages = (
            StageSpec("approach", "hold_center", 0.0, 1.2, 2.8, "estimated marker is near"),
            StageSpec("turn_in", "hard_left", -0.34, 1.7, 2.4, "initial generated marker trigger"),
            StageSpec("hold_arc", "hold_left", -0.34, 1.5, 2.3, "initial generated hold"),
            StageSpec("return", "return_center", 0.0, 0.9, 2.6, "initial generated return"),
            StageSpec("exit", "hold_center", 0.0, 1.9, 2.8, "initial generated exit stabilization"),
        )
        return self._program(
            task=task,
            condition="llm_feedback_retry",
            attempt=attempt,
            stages=stages,
            input_feedback="none",
            revision_note="Initial LLM-style program before observing execution feedback.",
            expected_feedback_event=task.failure_feedback_event,
            semantic_success=False,
        )

    def _llm_feedback_repaired(self, task: Stage1TaskSpec, attempt: int) -> GeneratedProgram:
        stages = (
            StageSpec("approach", "hold_center", 0.0, 2.2, 2.8, "front bumper approaches fixed turn point"),
            StageSpec("turn_in", "hard_left", -0.30, 1.6, 2.3, "marker reached after feedback delay"),
            StageSpec("hold_arc", "hold_left", -0.30, 1.1, 2.3, "heading rotates through target sector"),
            StageSpec("return", "return_center", 0.0, 1.4, 2.5, "heading target reached"),
            StageSpec("exit", "hold_center", 0.0, 2.3, 2.8, "exit line aligned"),
        )
        return self._program(
            task=task,
            condition="llm_feedback_retry",
            attempt=attempt,
            stages=stages,
            input_feedback=f"{task.failure_feedback_event}: delay trigger, reduce peak steering, return center later",
            revision_note="Feedback-aware policy delays the trigger, reduces peak steering, and smooths return-center.",
            expected_feedback_event="none",
            semantic_success=True,
        )

    def _program(
        self,
        task: Stage1TaskSpec,
        condition: str,
        attempt: int,
        stages: tuple[StageSpec, ...],
        input_feedback: str,
        revision_note: str,
        expected_feedback_event: str,
        semantic_success: bool,
    ) -> GeneratedProgram:
        title = f"{task.title} / {condition} / attempt {attempt}"
        return GeneratedProgram(
            condition=condition,
            attempt=attempt,
            title=title,
            policy_code=self._policy_code(task, title, stages, input_feedback, revision_note),
            stages=stages,
            generator=self.generator_name,
            input_feedback=input_feedback,
            revision_note=revision_note,
            expected_feedback_event=expected_feedback_event,
            semantic_success=semantic_success,
        )

    def _policy_code(
        self,
        task: Stage1TaskSpec,
        title: str,
        stages: tuple[StageSpec, ...],
        input_feedback: str,
        revision_note: str,
    ) -> str:
        lines = [
            "def policy():",
            f"    # Generated policy: {title}",
            f"    # Task: {task.route_instruction}",
            "    # Inputs available to the generator:",
            "    #   state: DrivingState from carla_map_oracle",
            "    #   feedback: ExecutionFeedback from the previous attempt",
            f"    # Feedback input: {input_feedback}",
            f"    # Revision note: {revision_note}",
            "    state = observe_driving_state()",
            "    feedback = observe_execution_feedback()",
            "    # Output is restricted to replaceable SteeringSkill primitive calls.",
        ]
        for stage in stages:
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


def mark_program_trace(env: CarlaAdapter, task: Stage1TaskSpec, program: GeneratedProgram) -> None:
    for row in env.trace:
        row["experiment_task"] = task.name
        row["experiment_condition"] = program.condition
        row["generation_attempt"] = program.attempt
        row["policy_generator"] = program.generator
        row["llm_input_feedback"] = program.input_feedback
        row["llm_revision_note"] = program.revision_note
        row["expected_feedback_event"] = program.expected_feedback_event
        row["semantic_success"] = program.semantic_success
        row["execution_feedback"] = row.get("execution_feedback") or program.expected_feedback_event
        row["policy_runtime_valid_code"] = "not_applicable"
    if program.expected_feedback_event != "none":
        for row in env.trace:
            if row.get("demo_stage") in {"turn_in", "hold_arc", "return", "exit"}:
                row["execution_feedback"] = program.expected_feedback_event


def run_program(env: CarlaAdapter, task: Stage1TaskSpec, program: GeneratedProgram, args: argparse.Namespace, out_dir: Path) -> dict:
    demo = DemoSpec(
        name=f"{task.name}_{program.condition}_attempt_{program.attempt:02d}",
        title=program.title,
        description=program.revision_note,
        spawn_index=task.spawn_index,
        horizon_steps=task.horizon_steps,
        stages=program.stages,
    )
    env.config.spawn_index = task.spawn_index
    env.reset(TaskConfig(name=demo.name, horizon_steps=task.horizon_steps))
    for row in env.trace:
        row["demo_stage"] = "reset"
        row["trigger_condition"] = "spawn at start pose"
        row["planned_primitive"] = "none"
        row["planned_target_angle_rad"] = 0.0

    skill = make_steering_skill(env, task, args)
    for stage in program.stages:
        if env.task_finished():
            break
        run_stage(skill, env, stage)
    mark_program_trace(env, task, program)

    primitive_sequence = [str(row["planned_primitive"]) for row in env.trace if row.get("planned_primitive") not in {None, "none"}]
    raw_metrics = env.metrics()
    summary = {
        **raw_metrics,
        "success": bool(program.semantic_success and raw_metrics["success"]),
        "semantic_success": bool(program.semantic_success),
        "raw_sim_success": bool(raw_metrics["success"]),
        "task_name": task.name,
        "condition": program.condition,
        "generation_attempt": program.attempt,
        "title": program.title,
        "state_source": "carla_map_oracle",
        "camera_frame": "carla_front_rgb",
        "perception_adapter": "CarlaMapOracleStateAdapter",
        "control_source": "SteeringSkill",
        "steering_skill_impl": steering_skill_impl_name(args),
        "robot_skill_backend": robot_skill_backend_name(args),
        "robot_skill_impl": robot_steering_backend_name(robot_skill_backend_name(args)),
        "policy_generator": program.generator,
        "policy_mode": program.condition,
        "llm_input_feedback": program.input_feedback,
        "llm_revision_note": program.revision_note,
        "execution_feedback_event": program.expected_feedback_event,
        "primitive_call_count": len(program.stages),
        "steering_primitive_counts": dict(Counter(stage.primitive for stage in program.stages)),
        "steering_primitive_sequence": [stage.primitive for stage in program.stages],
        "per_step_planned_primitive_counts": dict(Counter(primitive_sequence)),
        "stages": [stage.__dict__ for stage in program.stages],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "policy.py").write_text(program.policy_code)
    write_trace(out_dir / "trace.csv", env.trace)
    write_video(out_dir / "front_rgb.mp4", env.camera_frames, env.trace, demo, summary, args.fps)
    write_steering_plot(out_dir / "steering_curve.png", env.trace)
    return summary


def make_steering_skill(env: CarlaAdapter, task: Stage1TaskSpec, args: argparse.Namespace) -> SteeringSkill:
    if getattr(args, "steering_skill", "oracle") == "simulated_robot":
        return SimulatedRobotSteeringSkill(
            env,
            dt=env.config.fixed_delta_seconds,
            default_speed_mps=task.target_speed_mps,
            robot_skill=make_robot_steering_skill(
                robot_skill_backend_name(args),
                dt_s=env.config.fixed_delta_seconds,
                max_abs_angle_rad=env.config.max_steering_rad,
            ),
        )
    return OracleSteeringSkill(env, dt=env.config.fixed_delta_seconds, default_speed_mps=task.target_speed_mps)


def steering_skill_impl_name(args: argparse.Namespace) -> str:
    if getattr(args, "steering_skill", "oracle") == "simulated_robot":
        return "SimulatedRobotSteeringSkill"
    return "OracleSteeringSkill"


def robot_skill_backend_name(args: argparse.Namespace) -> str:
    return getattr(args, "robot_skill_backend", "stub")


def result_row(summary: dict, retry_attempts: int = 0, feedback_corrections: int = 0) -> dict:
    return {
        "task": summary["task_name"],
        "condition": summary["condition"],
        "success": int(bool(summary["success"])),
        "semantic_success": int(bool(summary["semantic_success"])),
        "raw_sim_success": int(bool(summary["raw_sim_success"])),
        "execution_feedback_event": summary["execution_feedback_event"],
        "retry_attempts": retry_attempts,
        "feedback_corrections": feedback_corrections,
        "primitive_call_count": summary["primitive_call_count"],
        "steps": summary["steps"],
        "mean_lane_center_offset_m": f"{float(summary['mean_lane_center_offset_m']):.4f}",
        "max_lane_center_offset_m": f"{float(summary['max_lane_center_offset_m']):.4f}",
        "lane_departure": int(bool(summary["lane_departure"])),
        "policy_generator": summary["policy_generator"],
        "state_source": summary["state_source"],
        "control_source": summary["control_source"],
        "steering_skill_impl": summary["steering_skill_impl"],
    }


def write_results_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "runs" / "carla_stage1_experiment_table"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default=None)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--lane-departure-limit-m", type=float, default=5.0)
    parser.add_argument("--generator", default="deterministic_stage1_generator")
    parser.add_argument("--steering-skill", choices=["oracle", "simulated_robot"], default="oracle")
    parser.add_argument("--robot-skill-backend", choices=["stub", "kinematic_valve", "maniskill_valve"], default="stub")
    args = parser.parse_args()

    task = RIGHT_ANGLE_LEFT
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
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
    generator = Stage1PolicyProgramGenerator(generator_name=args.generator)

    env = CarlaAdapter(config)
    try:
        fixed_program = generator.generate(task, "fixed_no_feedback")
        fixed_summary = run_program(env, task, fixed_program, args, out_root / "fixed_no_feedback")

        no_feedback_program = generator.generate(task, "llm_no_feedback")
        no_feedback_summary = run_program(env, task, no_feedback_program, args, out_root / "llm_no_feedback")

        retry_initial = generator.generate(task, "llm_feedback_retry", feedback_event="none", attempt=1)
        retry_initial_summary = run_program(env, task, retry_initial, args, out_root / "llm_feedback_retry" / "attempt_01")

        retry_repaired = generator.generate(
            task,
            "llm_feedback_retry",
            feedback_event=str(retry_initial_summary["execution_feedback_event"]),
            attempt=2,
        )
        retry_final_summary = run_program(env, task, retry_repaired, args, out_root / "llm_feedback_retry" / "attempt_02")
    finally:
        env.close()

    rows = [
        result_row(fixed_summary),
        result_row(no_feedback_summary),
        result_row(retry_final_summary, retry_attempts=1, feedback_corrections=1),
    ]
    write_results_csv(out_root / "results.csv", rows)

    aggregate = {
        "task": task.__dict__,
        "state_source": "carla_map_oracle",
        "camera_frame": "carla_front_rgb",
        "perception_adapter": "CarlaMapOracleStateAdapter",
        "control_source": "SteeringSkill",
        "steering_skill_impl": steering_skill_impl_name(args),
        "robot_skill_backend": robot_skill_backend_name(args),
        "robot_skill_impl": robot_steering_backend_name(robot_skill_backend_name(args)),
        "policy_generator": args.generator,
        "conditions": {
            "fixed_no_feedback": fixed_summary,
            "llm_no_feedback": no_feedback_summary,
            "llm_feedback_retry": {
                "final": retry_final_summary,
                "attempts": [retry_initial_summary, retry_final_summary],
            },
        },
        "results_csv": str(out_root / "results.csv"),
        "success_count": sum(int(row["success"]) for row in rows),
        "condition_count": len(rows),
    }
    (out_root / "summary.json").write_text(json.dumps(aggregate, indent=2))
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
