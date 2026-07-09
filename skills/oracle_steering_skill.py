from __future__ import annotations

import math

from envs.simulator_adapter import SimulatorAdapter
from skills.feedback import SkillResult


class OracleSteeringSkill:
    """CARLA/simulator oracle implementation of the SteeringSkill protocol.

    The public methods are the low-level skill contract used by generated
    policy code. They intentionally expose only target angle, duration, and
    speed-setting primitives so a later robot imitation-learning skill can
    replace this class behind the same calls.
    """

    def __init__(self, env: SimulatorAdapter, dt: float = 0.1, default_speed_mps: float = 8.0):
        self.env = env
        self.dt = dt
        self.target_speed_mps = default_speed_mps
        self.trace_primitive_name: str | None = None

    def set_target_speed(self, speed_mps: float) -> None:
        self.target_speed_mps = max(0.0, min(float(speed_mps), 12.0))

    def set_trace_primitive(self, primitive_name: str | None) -> None:
        self.trace_primitive_name = primitive_name

    def steer_to(self, target_angle: float, duration: float, *, ramp_s: float | None = None) -> SkillResult:
        return self._run("steer_to", target_angle, duration if ramp_s is None else ramp_s)

    def hold(self, angle: float, duration: float, *, hold_s: float | None = None) -> SkillResult:
        if hold_s is not None:
            duration = hold_s
        return self._run("hold_steering", angle, duration)

    def return_center(self, duration: float, *, ramp_s: float | None = None) -> SkillResult:
        return self._run("return_center", 0.0, duration if ramp_s is None else ramp_s)

    def execute_turn(self, direction: str, duration: float = 0.9) -> SkillResult:
        target = -0.34 if direction == "turn_left" or direction == "left" else 0.34
        return self._run("execute_turn", target, duration)

    def _run(self, name: str, target_angle: float, duration: float) -> SkillResult:
        steps = max(1, int(math.ceil(duration / self.dt)))
        primitive_name = self.trace_primitive_name or name
        start_angle = self.env.get_observation().steering_angle_rad
        if hasattr(self.env, "set_trace_context"):
            self.env.set_trace_context(name, primitive_name, target_angle)
        try:
            for idx in range(steps):
                if self.env.task_finished():
                    break
                if name in {"steer_to", "return_center", "execute_turn"}:
                    alpha = float(idx + 1) / float(steps)
                    command_angle = start_angle + (target_angle - start_angle) * alpha
                else:
                    command_angle = target_angle
                self.env.step(command_angle, self.target_speed_mps)
        finally:
            if hasattr(self.env, "clear_trace_context"):
                self.env.clear_trace_context()
        obs = self.env.get_observation()
        event = self.env.evaluate_after_skill(name, target_angle)
        success = event in {"none", "turn_too_early", "turn_too_late"}
        return SkillResult(
            name=name,
            success=success,
            target_angle=target_angle,
            final_angle=obs.steering_angle_rad,
            duration=steps * self.dt,
            event=event,
        )
