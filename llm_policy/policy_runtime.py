from __future__ import annotations

import math
from dataclasses import dataclass, field
from collections import Counter
from typing import Callable

from envs.simulator_adapter import SimulatorAdapter
from perception.oracle_state_adapter import OracleStateAdapter
from perception.perception_adapter import PerceptionAdapter
from perception.state_schema import DrivingState
from skills.discrete_steering import (
    HARD_LEFT,
    HARD_RIGHT,
    HOLD_CENTER,
    MEDIUM_LEFT,
    MEDIUM_RIGHT,
    SOFT_LEFT,
    SOFT_RIGHT,
    SteeringPrimitiveSpec,
)
from skills.feedback import ExecutionFeedback
from skills.steering_skill import SteeringSkill


@dataclass
class PolicyRuntimeStats:
    primitive_call_count: int = 0
    valid_code: bool = True
    feedback_recovery_count: int = 0
    low_confidence_actions: int = 0
    events: list[str] = field(default_factory=list)
    steering_primitive_counts: dict[str, int] = field(default_factory=dict)
    steering_primitive_sequence: list[str] = field(default_factory=list)


class RestrictedPolicyRuntime:
    def __init__(
        self,
        env: SimulatorAdapter,
        steering_skill: SteeringSkill,
        perception: PerceptionAdapter | None = None,
        use_oracle_state: bool = False,
        max_policy_iterations: int = 500,
    ):
        self.env = env
        self.steering_skill = steering_skill
        self.perception = perception
        self.oracle = OracleStateAdapter(env)
        self.use_oracle_state = use_oracle_state
        self.max_policy_iterations = max_policy_iterations
        self.feedback = ExecutionFeedback()
        self.stats = PolicyRuntimeStats()
        self._active_primitive_name: str | None = None

    def run(self, policy_code: str) -> PolicyRuntimeStats:
        env: dict[str, object] = {
            "__builtins__": {
                "abs": abs,
                "min": min,
                "max": max,
                "float": float,
                "int": int,
                "bool": bool,
                "range": range,
            },
            "math": math,
            "observe_driving_state": self.observe_driving_state,
            "observe_execution_feedback": self.observe_execution_feedback,
            "task_finished": self.task_finished,
            "steer_to": self.steer_to,
            "hold_steering": self.hold_steering,
            "return_center": self.return_center,
            "soft_left": self.soft_left,
            "medium_left": self.medium_left,
            "hard_left": self.hard_left,
            "soft_right": self.soft_right,
            "medium_right": self.medium_right,
            "hard_right": self.hard_right,
            "hold_center": self.hold_center,
            "repeat_with_stronger_steering": self.repeat_with_stronger_steering,
            "keep_lane_center": self.keep_lane_center,
            "recover_from_offset": self.recover_from_offset,
            "follow_front_vehicle": self.follow_front_vehicle,
            "prepare_turn": self.prepare_turn,
            "execute_turn": self.execute_turn,
            "slow_down": self.slow_down,
            "maintain_speed": self.maintain_speed,
        }
        try:
            exec(policy_code, env, env)
            policy = env.get("policy")
            if not callable(policy):
                raise ValueError("policy_code did not define callable policy()")
            policy()
        except Exception as exc:
            self.stats.valid_code = False
            self.stats.events.append(f"runtime_error:{type(exc).__name__}:{exc}")
        return self.stats

    def observe_driving_state(self) -> DrivingState:
        if self.use_oracle_state:
            return self.oracle.estimate_from_env()
        if self.perception is None:
            raise RuntimeError("perception adapter is required when use_oracle_state=False")
        return self.perception.estimate(self.env.get_observation())

    def observe_execution_feedback(self) -> ExecutionFeedback:
        return self.feedback

    def task_finished(self) -> bool:
        if self.stats.primitive_call_count >= self.max_policy_iterations:
            return True
        return self.env.task_finished()

    def steer_to(self, angle: float, duration: float) -> None:
        direct = self._active_primitive_name is None
        if direct:
            self._begin_primitive("steer_to")
        try:
            self._record_result(self.steering_skill.steer_to(angle, duration))
        finally:
            if direct:
                self._end_primitive()

    def hold_steering(self, angle: float, duration: float) -> None:
        direct = self._active_primitive_name is None
        if direct:
            self._begin_primitive("hold_steering")
        try:
            self._record_result(self.steering_skill.hold(angle, duration))
        finally:
            if direct:
                self._end_primitive()

    def return_center(self, duration: float) -> None:
        direct = self._active_primitive_name is None
        if direct:
            self._begin_primitive("return_center")
        try:
            self._record_result(self.steering_skill.return_center(duration))
        finally:
            if direct:
                self._end_primitive()

    def soft_left(self) -> None:
        self._run_discrete_primitive(SOFT_LEFT)

    def medium_left(self) -> None:
        self._run_discrete_primitive(MEDIUM_LEFT)

    def hard_left(self) -> None:
        self._run_discrete_primitive(HARD_LEFT)

    def soft_right(self) -> None:
        self._run_discrete_primitive(SOFT_RIGHT)

    def medium_right(self) -> None:
        self._run_discrete_primitive(MEDIUM_RIGHT)

    def hard_right(self) -> None:
        self._run_discrete_primitive(HARD_RIGHT)

    def hold_center(self) -> None:
        self._run_discrete_primitive(HOLD_CENTER)

    def repeat_with_stronger_steering(self) -> None:
        target = self.feedback.target_angle
        if target is None or abs(target) < 0.08:
            lane_error = self.feedback.lane_error_after_skill_m or 0.0
            heading_error = self.feedback.heading_error_after_skill_rad or 0.0
            if lane_error > 0.45 or heading_error > 0.10:
                self.hard_left()
            elif lane_error < -0.45 or heading_error < -0.10:
                self.hard_right()
            elif lane_error > 0.18 or heading_error > 0.04:
                self.medium_left()
            elif lane_error < -0.18 or heading_error < -0.04:
                self.medium_right()
            else:
                self.hold_center()
            return
        if target < 0:
            if abs(target) < 0.16:
                self.medium_left()
            else:
                self.hard_left()
        else:
            if abs(target) < 0.16:
                self.medium_right()
            else:
                self.hard_right()

    def keep_lane_center(self) -> None:
        state = self.observe_driving_state()
        offset = state.lane_center_offset_m.as_float()
        heading = state.heading_error_rad.as_float()
        curvature = state.lane_curvature.as_float()
        if offset > 0.35 and heading < -0.10:
            self.hold_center()
        elif offset < -0.35 and heading > 0.10:
            self.hold_center()
        elif offset > 0.45 and heading > 0.04:
            self.hard_left()
        elif offset < -0.45 and heading < -0.04:
            self.hard_right()
        elif offset > 0.90:
            self.hard_left()
        elif offset < -0.90:
            self.hard_right()
        elif offset > 0.35:
            self.medium_left()
        elif offset < -0.35:
            self.medium_right()
        elif curvature > 0.015:
            self.medium_right()
        elif curvature > 0.006:
            self.soft_right()
        elif curvature < -0.015:
            self.medium_left()
        elif curvature < -0.006:
            self.soft_left()
        elif heading > 0.18:
            self.medium_left()
        elif heading < -0.18:
            self.medium_right()
        elif offset > 0.15 or heading > 0.06:
            self.soft_left()
        elif offset < -0.15 or heading < -0.06:
            self.soft_right()
        else:
            self.hold_center()

    def recover_from_offset(self) -> None:
        state = self.observe_driving_state()
        offset = state.lane_center_offset_m.as_float()
        heading = state.heading_error_rad.as_float()
        if offset > 0.35 and heading < -0.10:
            self.hold_center()
        elif offset < -0.35 and heading > 0.10:
            self.hold_center()
        elif offset > 0.45 and heading > 0.04:
            self.hard_left()
        elif offset < -0.45 and heading < -0.04:
            self.hard_right()
        elif offset > 0.90:
            self.hard_left()
        elif offset < -0.90:
            self.hard_right()
        elif offset > 0.35:
            self.medium_left()
        elif offset < -0.35:
            self.medium_right()
        elif heading > 0.18:
            self.medium_left()
        elif heading < -0.18:
            self.medium_right()
        elif offset > 0.15 or heading > 0.06:
            self.soft_left()
        elif offset < -0.15 or heading < -0.06:
            self.soft_right()
        else:
            self.hold_center()

    def follow_front_vehicle(self) -> None:
        state = self.observe_driving_state()
        dist = state.front_vehicle_distance_m.as_float(20.0)
        current_speed = state.speed_mps.as_float(6.0)
        target_speed = 4.0 if dist < 8.0 else min(current_speed, 5.5)
        self.steering_skill.set_target_speed(target_speed)
        self.stats.feedback_recovery_count += 1
        self.keep_lane_center()

    def prepare_turn(self, direction: str) -> None:
        self.steering_skill.set_target_speed(5.8)
        self.keep_lane_center()

    def execute_turn(self, direction: str) -> None:
        self.steering_skill.set_target_speed(5.2)
        self._begin_primitive(f"execute_{direction}")
        try:
            self._record_result(self.steering_skill.execute_turn(direction, duration=0.85))
        finally:
            self._end_primitive()

    def slow_down(self) -> None:
        self.steering_skill.set_target_speed(3.5)
        self.stats.low_confidence_actions += 1
        self.return_center(duration=0.25)

    def maintain_speed(self, speed_mps: float = 8.0) -> None:
        self.steering_skill.set_target_speed(speed_mps)
        self.keep_lane_center()

    def _run_discrete_primitive(self, spec: SteeringPrimitiveSpec) -> None:
        self._begin_primitive(spec.name)
        try:
            self.steer_to(spec.angle_rad, duration=spec.steer_duration_s)
            if spec.hold_duration_s > 0:
                self.hold_steering(spec.angle_rad, duration=spec.hold_duration_s)
            if spec.return_duration_s > 0:
                self.return_center(duration=spec.return_duration_s)
        finally:
            self._end_primitive()

    def _begin_primitive(self, name: str) -> None:
        self._active_primitive_name = name
        self.stats.steering_primitive_sequence.append(name)
        counts = Counter(self.stats.steering_primitive_counts)
        counts[name] += 1
        self.stats.steering_primitive_counts = dict(counts)
        self.steering_skill.set_trace_primitive(name)

    def _end_primitive(self) -> None:
        self._active_primitive_name = None
        self.steering_skill.set_trace_primitive(None)

    def _record_result(self, result) -> None:
        oracle = self.env.get_oracle_state_values()
        self.feedback = ExecutionFeedback.from_result(
            result,
            lane_error_after_skill_m=float(oracle["lane_center_offset_m"] or 0.0),
            heading_error_after_skill_rad=float(oracle["heading_error_rad"] or 0.0),
        )
        self.stats.primitive_call_count += 1
        if result.event and result.event != "none":
            self.stats.events.append(result.event)
