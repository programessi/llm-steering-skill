from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.carla_adapter import CarlaAdapter, CarlaAdapterConfig
from envs.simulator_adapter import TaskConfig
from feedback.trace_feedback_adapter import (
    apply_auto_feedback_to_trace,
    feedback_input_text,
    infer_execution_feedback,
)
from experiments.run_carla_driving_test_demos import (
    DemoSpec,
    StageSpec,
    run_stage,
    write_steering_plot,
    write_video,
)
from experiments.run_carla_feedback_retry_demo import write_trace
from experiments.run_carla_stage1_experiment_table import (
    RIGHT_ANGLE_LEFT,
    GeneratedProgram,
    Stage1PolicyProgramGenerator,
    Stage1TaskSpec,
    STAGE1_TASKS,
    result_row,
    write_results_csv,
)
from llm_policy.llm_code_generator import OpenAICompatiblePolicyGenerator
from perception.model_backed_carla_adapter import ModelBackedCarlaPerceptionAdapter
from perception.oracle_state_adapter import OracleStateAdapter
from perception.state_schema import estimated
from perception.trilitenet_lane_adapter import TriLiteNetLanePerceptionAdapter
from robot_steering.backends import make_robot_steering_skill, robot_steering_backend_name
from skills.feedback import ExecutionFeedback
from skills.oracle_steering_skill import OracleSteeringSkill
from skills.simulated_robot_steering_skill import SimulatedRobotSteeringSkill
from skills.steering_skill import SteeringSkill


CONDITIONAL_POLICY_TEMPLATE = '''\
def policy():
    # Generated conditional policy: {title}
    # Task: {task}
    # Inputs:
    #   state = observe_driving_state()
    #   feedback = observe_execution_feedback()
    # Output:
    #   conditional calls to execute_primitive(...). The policy never sends
    #   continuous wheel commands directly.
    params = {{
        "trigger_distance_m": 5.2,
        "turn_angle_rad": -0.34,
        "turn_duration_s": 1.7,
        "hold_duration_s": 1.5,
        "return_duration_s": 0.9,
        "approach_step_s": 0.25,
    }}

    feedback = observe_execution_feedback()
    if feedback.event == "turn_too_early":
        params["trigger_distance_m"] -= 1.45
        params["turn_angle_rad"] *= 0.75
        params["turn_duration_s"] -= 0.3
        params["hold_duration_s"] -= 0.8
        params["return_duration_s"] += 0.6
    elif feedback.event == "turn_too_late":
        params["trigger_distance_m"] += 1.0
        params["turn_angle_rad"] *= 1.08

    while not task_finished():
        state = observe_driving_state()
        distance = state.distance_to_maneuver_m.as_float(0.0)
        if distance > params["trigger_distance_m"]:
            execute_primitive(
                name="hold_center",
                target_angle_rad=0.0,
                duration_s=params["approach_step_s"],
                speed_mps=2.8,
                stage="approach_wait",
                trigger="distance_to_maneuver_m > trigger_distance_m",
            )
            continue

        execute_primitive(
            name="hard_left",
            target_angle_rad=params["turn_angle_rad"],
            duration_s=params["turn_duration_s"],
            speed_mps=2.4,
            stage="turn_in",
            trigger="distance_to_maneuver_m <= trigger_distance_m",
        )
        execute_primitive(
            name="hold_left",
            target_angle_rad=params["turn_angle_rad"],
            duration_s=params["hold_duration_s"],
            speed_mps=2.3,
            stage="hold_arc",
            trigger="heading rotates through target sector",
        )
        execute_primitive(
            name="return_center",
            target_angle_rad=0.0,
            duration_s=params["return_duration_s"],
            speed_mps=2.5,
            stage="return",
            trigger="heading target reached",
        )
        execute_primitive(
            name="hold_center",
            target_angle_rad=0.0,
            duration_s=2.2,
            speed_mps=2.8,
            stage="exit",
            trigger="exit line aligned",
        )
        mark_task_complete()
'''


@dataclass
class ExecutedPolicyStats:
    valid_code: bool = True
    primitive_call_count: int = 0
    events: list[str] = field(default_factory=list)
    steering_primitive_counts: dict[str, int] = field(default_factory=dict)
    steering_primitive_sequence: list[str] = field(default_factory=list)


class CarlaConditionalPolicyRuntime:
    def __init__(
        self,
        env: CarlaAdapter,
        skill: SteeringSkill,
        feedback: ExecutionFeedback,
        route_command: str = "turn_left",
        start_distance_to_maneuver_m: float = 8.0,
        max_policy_iterations: int = 80,
        perception_source: str = "oracle",
        perception_adapter: ModelBackedCarlaPerceptionAdapter | None = None,
    ):
        self.env = env
        self.skill = skill
        self.feedback = feedback
        self.route_command = route_command
        self.oracle = OracleStateAdapter(env)
        self.perception_source = perception_source
        self.perception_adapter = perception_adapter
        self.start_distance_to_maneuver_m = float(start_distance_to_maneuver_m)
        self.max_policy_iterations = max_policy_iterations
        self.stats = ExecutedPolicyStats()
        self._task_complete = False
        self.last_perceived_state = None
        self.last_oracle_state = None

    def run(self, policy_code: str) -> ExecutedPolicyStats:
        scope: dict[str, object] = {
            "__builtins__": {
                "abs": abs,
                "min": min,
                "max": max,
                "float": float,
                "int": int,
                "bool": bool,
                "range": range,
            },
            "observe_driving_state": self.observe_driving_state,
            "observe_execution_feedback": self.observe_execution_feedback,
            "execute_primitive": self.execute_primitive,
            "task_finished": self.task_finished,
            "mark_task_complete": self.mark_task_complete,
        }
        try:
            exec(policy_code, scope, scope)
            policy = scope.get("policy")
            if not callable(policy):
                raise ValueError("policy_code did not define callable policy()")
            policy()
        except Exception as exc:
            self.stats.valid_code = False
            self.stats.events.append(f"runtime_error:{type(exc).__name__}:{exc}")
        return self.stats

    def observe_driving_state(self):
        oracle_state = self.oracle.estimate_from_env()
        oracle_distance = max(0.0, self.start_distance_to_maneuver_m - self._distance_traveled_m())
        oracle_state.distance_to_maneuver_m = estimated(oracle_distance, 1.0, self.env.time_s)
        oracle_state.route_command = estimated(self.route_command, 1.0, self.env.time_s)
        self.last_oracle_state = oracle_state
        if self.perception_source in MODEL_MARKER_SOURCES:
            if self.perception_adapter is None:
                raise RuntimeError(f"{self.perception_source} perception_source requires a perception_adapter")
            state = self.perception_adapter.estimate(self.env.get_observation())
            state.route_command = estimated(self.route_command, 1.0, self.env.time_s)
        else:
            state = oracle_state
        self.last_perceived_state = state
        return state

    def observe_execution_feedback(self) -> ExecutionFeedback:
        return self.feedback

    def execute_primitive(
        self,
        name: str,
        target_angle_rad: float,
        duration_s: float,
        speed_mps: float,
        stage: str = "policy_step",
        trigger: str = "policy condition",
    ) -> None:
        if self.task_finished():
            return
        start_idx = len(self.env.trace)
        stage_spec = StageSpec(
            name=stage,
            primitive=name,
            target_angle_rad=float(target_angle_rad),
            duration_s=max(0.05, float(duration_s)),
            speed_mps=max(0.0, float(speed_mps)),
            trigger=trigger,
        )
        run_stage(self.skill, self.env, stage_spec)
        self._annotate_runtime_distance(start_idx)
        self._annotate_perception_state(start_idx)
        self.stats.primitive_call_count += 1
        self.stats.steering_primitive_sequence.append(name)
        counts = Counter(self.stats.steering_primitive_counts)
        counts[name] += 1
        self.stats.steering_primitive_counts = dict(counts)

    def mark_task_complete(self) -> None:
        self._task_complete = True

    def task_finished(self) -> bool:
        if self._task_complete:
            return True
        if self.stats.primitive_call_count >= self.max_policy_iterations:
            return True
        return self.env.task_finished()

    def _distance_traveled_m(self) -> float:
        rows = [row for row in self.env.trace if row.get("event") != "reset"]
        if not rows:
            return 0.0
        return float(sum(float(row.get("speed_mps") or 0.0) * self.env.config.fixed_delta_seconds for row in rows))

    def _annotate_runtime_distance(self, start_idx: int = 0) -> None:
        traveled_m = 0.0
        distances: list[float] = []
        for row in self.env.trace:
            if row.get("event") != "reset":
                traveled_m += float(row.get("speed_mps") or 0.0) * self.env.config.fixed_delta_seconds
            distances.append(max(0.0, self.start_distance_to_maneuver_m - traveled_m))
        for idx in range(max(0, start_idx), len(self.env.trace)):
            self.env.trace[idx]["runtime_distance_to_maneuver_m"] = distances[idx]
            self.env.trace[idx]["oracle_distance_to_maneuver_m"] = distances[idx]

    def _annotate_perception_state(self, start_idx: int = 0) -> None:
        state = self.last_perceived_state
        oracle = self.last_oracle_state
        debug = getattr(self.perception_adapter, "last_debug", None)
        lane_debug = getattr(getattr(self.perception_adapter, "lane_adapter", None), "last_debug", None)
        if state is None:
            return
        perceived_distance = state.distance_to_maneuver_m.value
        oracle_distance = oracle.distance_to_maneuver_m.value if oracle is not None else None
        perceived_offset = state.lane_center_offset_m.value
        perceived_heading = state.heading_error_rad.value
        oracle_offset = oracle.lane_center_offset_m.value if oracle is not None else None
        oracle_heading = oracle.heading_error_rad.value if oracle is not None else None
        distance_error = None
        if perceived_distance is not None and oracle_distance is not None:
            distance_error = float(perceived_distance) - float(oracle_distance)
        offset_error = None
        if perceived_offset is not None and oracle_offset is not None:
            offset_error = float(perceived_offset) - float(oracle_offset)
        heading_error = None
        if perceived_heading is not None and oracle_heading is not None:
            heading_error = float(perceived_heading) - float(oracle_heading)
        for idx in range(max(0, start_idx), len(self.env.trace)):
            row = self.env.trace[idx]
            row["llm_perception_source"] = self.perception_source
            row["perceived_distance_to_maneuver_m"] = perceived_distance
            row["perceived_lane_center_offset_m"] = perceived_offset
            row["perceived_heading_error_rad"] = perceived_heading
            row["perceived_perception_confidence"] = state.perception_confidence.value
            row["oracle_distance_to_maneuver_m"] = oracle_distance
            row["oracle_lane_center_offset_m"] = oracle_offset
            row["oracle_heading_error_rad"] = oracle_heading
            row["perceived_distance_error_m"] = distance_error
            row["perceived_lane_center_offset_error_m"] = offset_error
            row["perceived_heading_error_rad_error"] = heading_error
            if debug is not None:
                row["marker_detected"] = debug.marker_detected
                row["marker_class_name"] = debug.marker_class_name
                row["marker_confidence"] = debug.marker_confidence
                row["marker_distance_source"] = debug.marker_distance_source
            if lane_debug is not None:
                row["lane_model_source"] = lane_debug.source
                row["lane_model_config"] = lane_debug.model_config
                row["lane_model_lane_pixel_count"] = lane_debug.lane_pixel_count
                row["lane_model_drivable_pixel_count"] = lane_debug.drivable_pixel_count
                row["lane_model_confidence"] = lane_debug.lane_confidence
                row["lane_model_drivable_confidence"] = lane_debug.drivable_confidence
                row["lane_model_px_per_m"] = lane_debug.px_per_m


class ConditionalPolicyProgramGenerator(Stage1PolicyProgramGenerator):
    def generate_conditional(
        self,
        task: Stage1TaskSpec,
        condition: str,
        feedback_event: str = "none",
        attempt: int = 1,
    ) -> GeneratedProgram:
        has_feedback = feedback_event not in {"none", "", None}
        title = f"{task.title} / {condition} / attempt {attempt}"
        policy_code = CONDITIONAL_POLICY_TEMPLATE.format(title=title, task=task.route_instruction)
        expected_feedback_event = "pending_auto_feedback"
        input_feedback = feedback_input_text(ExecutionFeedback(event=feedback_event))
        revision_note = (
            "Conditional policy reads ExecutionFeedback and changes trigger/angle/duration at runtime."
            if has_feedback
            else "Conditional policy uses the initial parameters because no failure feedback is available yet."
        )
        return GeneratedProgram(
            condition=condition,
            attempt=attempt,
            title=title,
            policy_code=policy_code,
            stages=(),
            generator=self.generator_name,
            input_feedback=input_feedback,
            revision_note=revision_note,
            expected_feedback_event=expected_feedback_event,
            semantic_success=has_feedback,
        )


class LLMConditionalPolicyProgramGenerator(ConditionalPolicyProgramGenerator):
    def __init__(self, generator_name: str = "openai_compatible_llm_generator"):
        super().__init__(generator_name=generator_name)
        self.llm = OpenAICompatiblePolicyGenerator()

    def generate_conditional(
        self,
        task: Stage1TaskSpec,
        condition: str,
        feedback_event: str = "none",
        attempt: int = 1,
    ) -> GeneratedProgram:
        has_feedback = feedback_event not in {"none", "", None}
        title = f"{task.title} / {condition} / attempt {attempt}"
        input_feedback = feedback_input_text(ExecutionFeedback(event=feedback_event))
        previous_failure = (
            input_feedback
            if has_feedback
            else "none"
        )
        result = self.llm.generate(
            task.route_instruction,
            feedback_event=feedback_event,
            previous_failure=previous_failure,
        )
        expected_feedback_event = "pending_auto_feedback"
        revision_note = (
            f"Generated online by {result.model} through OpenAI-compatible chat completions; "
            f"AST validated; repair_attempts={result.repair_attempts}."
        )
        return GeneratedProgram(
            condition=condition,
            attempt=attempt,
            title=title,
            policy_code=result.code,
            stages=(),
            generator=self.generator_name,
            input_feedback=input_feedback,
            revision_note=revision_note,
            expected_feedback_event=expected_feedback_event,
            semantic_success=has_feedback,
        )


def mark_trace(env: CarlaAdapter, task: Stage1TaskSpec, program: GeneratedProgram, stats: ExecutedPolicyStats) -> None:
    for row in env.trace:
        row["experiment_task"] = task.name
        row["experiment_condition"] = program.condition
        row["generation_attempt"] = program.attempt
        row["policy_generator"] = program.generator
        row["llm_input_feedback"] = program.input_feedback
        row["llm_revision_note"] = program.revision_note
        row["expected_feedback_event"] = program.expected_feedback_event
        row["semantic_success"] = program.semantic_success
        row["policy_runtime_valid_code"] = stats.valid_code
        row["execution_feedback"] = row.get("execution_feedback") or "pending_auto_feedback"


def run_conditional_program(
    env: CarlaAdapter,
    task: Stage1TaskSpec,
    program: GeneratedProgram,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict:
    demo = DemoSpec(
        name=f"{task.name}_{program.condition}_attempt_{program.attempt:02d}",
        title=program.title,
        description=program.revision_note,
        spawn_index=task.spawn_index,
        horizon_steps=task.horizon_steps,
        stages=(),
    )
    env.config.spawn_index = task.spawn_index
    env.reset(TaskConfig(name=demo.name, horizon_steps=task.horizon_steps))
    if args.perception_source in MODEL_MARKER_SOURCES:
        marker_bp = env.spawn_visual_maneuver_marker(
            distance_m=args.visual_marker_distance_m,
            lateral_offset_m=args.visual_marker_lateral_offset_m,
            blueprint_id=args.visual_marker_blueprint,
        )
    else:
        marker_bp = None
    for row in env.trace:
        row["demo_stage"] = "reset"
        row["trigger_condition"] = "spawn at start pose"
        row["planned_primitive"] = "none"
        row["planned_target_angle_rad"] = 0.0

    skill = make_steering_skill(env, task, args)
    perception_adapter = make_perception_adapter(env, task, args)
    feedback = ExecutionFeedback(event=program.input_feedback.split(":")[0] if program.input_feedback != "none" else "none")
    runtime = CarlaConditionalPolicyRuntime(
        env=env,
        skill=skill,
        feedback=feedback,
        route_command=route_command_for_task(task),
        start_distance_to_maneuver_m=args.start_distance_to_maneuver_m,
        max_policy_iterations=args.max_policy_iterations,
        perception_source=args.perception_source,
        perception_adapter=perception_adapter,
    )
    stats = runtime.run(program.policy_code)
    mark_trace(env, task, program, stats)
    raw_metrics = env.metrics()
    auto_feedback = infer_execution_feedback(
        env.trace,
        raw_metrics,
        policy_runtime_valid_code=stats.valid_code,
        max_lane_error_m=feedback_lane_error_limit_m(task),
    )
    apply_auto_feedback_to_trace(env.trace, auto_feedback)
    semantic_success = (auto_feedback.event or "none") == "none"
    summary = {
        **raw_metrics,
        "success": bool(semantic_success and raw_metrics["success"] and stats.valid_code),
        "semantic_success": bool(semantic_success),
        "raw_sim_success": bool(raw_metrics["success"]),
        "task_name": task.name,
        "condition": program.condition,
        "generation_attempt": program.attempt,
        "title": program.title,
        "state_source": state_source_name(args),
        "camera_frame": "carla_front_rgb",
        "perception_adapter": perception_adapter_name(args),
        "visual_marker_blueprint": marker_bp,
        "visual_marker_class": args.visual_marker_class,
        "visual_marker_distance_m": args.visual_marker_distance_m if args.perception_source in MODEL_MARKER_SOURCES else None,
        "control_source": "ExecutableConditionalPolicyRuntime->SteeringSkill",
        "steering_skill_impl": steering_skill_impl_name(args),
        "robot_skill_backend": robot_skill_backend_name(args),
        "robot_skill_impl": robot_steering_backend_name(robot_skill_backend_name(args)),
        "policy_generator": program.generator,
        "policy_mode": program.condition,
        "policy_runtime_valid_code": stats.valid_code,
        "runtime_events": stats.events,
        "llm_input_feedback": program.input_feedback,
        "llm_revision_note": program.revision_note,
        "feedback_source": "auto_trace_metrics",
        "execution_feedback_event": auto_feedback.event or "none",
        "execution_feedback": auto_feedback.to_dict(),
        "expected_feedback_event": program.expected_feedback_event,
        "primitive_call_count": stats.primitive_call_count,
        "steering_primitive_counts": stats.steering_primitive_counts,
        "steering_primitive_sequence": stats.steering_primitive_sequence,
        "start_distance_to_maneuver_m": args.start_distance_to_maneuver_m,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "policy.py").write_text(program.policy_code)
    write_trace(out_dir / "trace.csv", env.trace)
    write_video(out_dir / "front_rgb.mp4", env.camera_frames, env.trace, demo, summary, args.fps)
    write_steering_plot(out_dir / "steering_curve.png", env.trace)
    return summary


def route_command_for_task(task: Stage1TaskSpec) -> str:
    if task.name == "lane_change_left":
        return "lane_change_left"
    return "turn_left"


def feedback_lane_error_limit_m(task: Stage1TaskSpec) -> float:
    if task.name == "lane_change_left":
        return 2.2
    return 1.8


def make_steering_skill(env: CarlaAdapter, task: Stage1TaskSpec, args: argparse.Namespace):
    skill_name = getattr(args, "steering_skill", "oracle")
    if skill_name == "simulated_robot":
        robot_backend = robot_skill_backend_name(args)
        return SimulatedRobotSteeringSkill(
            env,
            dt=env.config.fixed_delta_seconds,
            default_speed_mps=task.target_speed_mps,
            robot_skill=make_robot_steering_skill(
                robot_backend,
                dt_s=env.config.fixed_delta_seconds,
                max_abs_angle_rad=env.config.max_steering_rad,
            ),
        )
    return OracleSteeringSkill(env, dt=env.config.fixed_delta_seconds, default_speed_mps=task.target_speed_mps)


def make_perception_adapter(env: CarlaAdapter, task: Stage1TaskSpec, args: argparse.Namespace):
    perception_source = getattr(args, "perception_source", "oracle")
    if perception_source not in MODEL_MARKER_SOURCES:
        return None
    lane_adapter = None
    if perception_source == "trilitenet_lane_yolo_marker":
        lane_adapter = TriLiteNetLanePerceptionAdapter(
            trilitenet_root=args.trilitenet_root,
            model_config=args.trilitenet_config,
            weights_path=args.trilitenet_weights,
            device=args.trilitenet_device,
            input_size=args.trilitenet_input_size,
            route_command=route_command_for_task(task),
        )
    return ModelBackedCarlaPerceptionAdapter(
        model_path=args.yolo_model,
        marker_class=args.visual_marker_class,
        marker_real_height_m=args.visual_marker_real_height_m,
        camera_fov_deg=env.config.fov,
        confidence_threshold=args.visual_marker_confidence,
        route_command=route_command_for_task(task),
        lane_adapter=lane_adapter,
    )


def state_source_name(args: argparse.Namespace) -> str:
    if getattr(args, "perception_source", "oracle") == "trilitenet_lane_yolo_marker":
        return "carla_front_rgb_trilitenet_lane_yolo_marker_distance"
    if getattr(args, "perception_source", "oracle") == "model_marker":
        return "carla_front_rgb_yolo_marker_distance"
    return "carla_map_oracle_with_fixed_point_distance"


def perception_adapter_name(args: argparse.Namespace) -> str:
    if getattr(args, "perception_source", "oracle") == "trilitenet_lane_yolo_marker":
        return "ModelBackedCarlaPerceptionAdapter(TriLiteNetLanePerceptionAdapter+YOLOMarker)"
    if getattr(args, "perception_source", "oracle") == "model_marker":
        return "ModelBackedCarlaPerceptionAdapter"
    return "CarlaMapOracleStateAdapter"


def steering_skill_impl_name(args: argparse.Namespace) -> str:
    if getattr(args, "steering_skill", "oracle") == "simulated_robot":
        return "SimulatedRobotSteeringSkill"
    return "OracleSteeringSkill"


def robot_skill_backend_name(args: argparse.Namespace) -> str:
    return getattr(args, "robot_skill_backend", "stub")


MODEL_MARKER_SOURCES = {"model_marker", "trilitenet_lane_yolo_marker"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "runs" / "carla_stage1_conditional_policy"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default=None)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--lane-departure-limit-m", type=float, default=5.0)
    parser.add_argument("--generator", default="deterministic_stage1_conditional_generator")
    parser.add_argument("--policy-generator", choices=["deterministic", "llm"], default="deterministic")
    parser.add_argument("--llm-fallback-deterministic", action="store_true")
    parser.add_argument("--task", choices=sorted(STAGE1_TASKS), default=RIGHT_ANGLE_LEFT.name)
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
    parser.add_argument("--start-distance-to-maneuver-m", type=float, default=8.0)
    parser.add_argument("--max-policy-iterations", type=int, default=80)
    args = parser.parse_args()

    task = STAGE1_TASKS[args.task]
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
    if args.policy_generator == "llm":
        try:
            generator = LLMConditionalPolicyProgramGenerator(generator_name="openai_compatible_llm_generator")
        except Exception:
            if not args.llm_fallback_deterministic:
                raise
            generator = ConditionalPolicyProgramGenerator(generator_name=args.generator)
    else:
        generator = ConditionalPolicyProgramGenerator(generator_name=args.generator)
    static_generator = Stage1PolicyProgramGenerator(generator_name="deterministic_stage1_generator")

    env = CarlaAdapter(config)
    try:
        static_no_feedback = static_generator.generate(task, "llm_no_feedback")
        static_no_feedback_summary = run_static_program(env, task, static_no_feedback, args, out_root / "static_sequence_no_feedback")

        static_initial = static_generator.generate(task, "llm_feedback_retry", feedback_event="none", attempt=1)
        static_initial_summary = run_static_program(env, task, static_initial, args, out_root / "static_sequence_feedback_retry" / "attempt_01")
        static_repaired = static_generator.generate(
            task,
            "llm_feedback_retry",
            feedback_event=str(static_initial_summary["execution_feedback_event"]),
            attempt=2,
        )
        static_repaired_summary = run_static_program(env, task, static_repaired, args, out_root / "static_sequence_feedback_retry" / "attempt_02")

        conditional_initial = generator.generate_conditional(task, "conditional_policy_feedback_retry", feedback_event="none", attempt=1)
        conditional_initial_summary = run_conditional_program(
            env,
            task,
            conditional_initial,
            args,
            out_root / "conditional_policy_feedback_retry" / "attempt_01",
        )
        conditional_repaired = generator.generate_conditional(
            task,
            "conditional_policy_feedback_retry",
            feedback_event=str(conditional_initial_summary["execution_feedback_event"]),
            attempt=2,
        )
        conditional_repaired_summary = run_conditional_program(
            env,
            task,
            conditional_repaired,
            args,
            out_root / "conditional_policy_feedback_retry" / "attempt_02",
        )
    finally:
        env.close()

    rows = [
        result_row(static_no_feedback_summary),
        result_row(static_repaired_summary, retry_attempts=1, feedback_corrections=1),
        result_row(conditional_repaired_summary, retry_attempts=1, feedback_corrections=1),
    ]
    rows[0]["condition"] = "static_sequence_no_feedback"
    rows[1]["condition"] = "static_sequence_feedback_retry"
    rows[2]["condition"] = "conditional_policy_feedback_retry"
    write_results_csv(out_root / "results.csv", rows)

    aggregate = {
        "task": task.__dict__,
        "state_source": state_source_name(args),
        "camera_frame": "carla_front_rgb",
        "perception_adapter": perception_adapter_name(args),
        "control_source": "ExecutableConditionalPolicyRuntime->SteeringSkill",
        "steering_skill_impl": steering_skill_impl_name(args),
        "robot_skill_backend": robot_skill_backend_name(args),
        "robot_skill_impl": robot_steering_backend_name(robot_skill_backend_name(args)),
        "conditions": {
            "static_sequence_no_feedback": static_no_feedback_summary,
            "static_sequence_feedback_retry": {
                "final": static_repaired_summary,
                "attempts": [static_initial_summary, static_repaired_summary],
            },
            "conditional_policy_feedback_retry": {
                "final": conditional_repaired_summary,
                "attempts": [conditional_initial_summary, conditional_repaired_summary],
            },
        },
        "results_csv": str(out_root / "results.csv"),
        "success_count": sum(int(row["success"]) for row in rows),
        "condition_count": len(rows),
    }
    (out_root / "summary.json").write_text(json.dumps(aggregate, indent=2))
    print(json.dumps(aggregate, indent=2))


def run_static_program(
    env: CarlaAdapter,
    task: Stage1TaskSpec,
    program: GeneratedProgram,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict:
    from experiments.run_carla_stage1_experiment_table import run_program

    return run_program(env, task, program, args, out_dir)


if __name__ == "__main__":
    main()
