from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class SkillResult:
    name: str
    success: bool
    target_angle: float | None
    final_angle: float | None
    duration: float
    event: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionFeedback:
    last_skill_name: str | None = None
    last_skill_success: bool = True
    target_angle: float | None = None
    actual_steering_angle: float | None = None
    lane_error_after_skill_m: float | None = None
    heading_error_after_skill_rad: float | None = None
    event: str | None = "none"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_result(
        cls,
        result: SkillResult,
        lane_error_after_skill_m: float,
        heading_error_after_skill_rad: float,
    ) -> "ExecutionFeedback":
        return cls(
            last_skill_name=result.name,
            last_skill_success=result.success,
            target_angle=result.target_angle,
            actual_steering_angle=result.final_angle,
            lane_error_after_skill_m=lane_error_after_skill_m,
            heading_error_after_skill_rad=heading_error_after_skill_rad,
            event=result.event,
        )

