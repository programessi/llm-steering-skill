from __future__ import annotations

from typing import Protocol

from skills.feedback import SkillResult


class SteeringSkill(Protocol):
    """Replaceable low-level steering skill boundary.

    Stage-1 policy code is allowed to choose a target steering angle and a
    duration, but it must not implement continuous wheel control directly.
    The oracle implementation drives CARLA. Stage 2 can replace it with an
    imitation-learned robot wheel/valve skill without changing policy code.
    """

    def set_target_speed(self, speed_mps: float) -> None:
        ...

    def set_trace_primitive(self, primitive_name: str | None) -> None:
        ...

    def steer_to(self, target_angle: float, duration: float, *, ramp_s: float | None = None) -> SkillResult:
        ...

    def hold(self, angle: float, duration: float, *, hold_s: float | None = None) -> SkillResult:
        ...

    def return_center(self, duration: float, *, ramp_s: float | None = None) -> SkillResult:
        ...

    def execute_turn(self, direction: str, duration: float = 0.9) -> SkillResult:
        ...
