from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class SteeringSkillCommand:
    skill_name: str
    current_angle_rad: float
    target_angle_rad: float
    duration_s: float
    hold_s: float = 0.0
    max_speed_rad_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SteeringObservation:
    timestamp_s: float
    wheel_angle_rad: float
    target_angle_rad: float
    wrist_image_path: str | None = None
    external_image_path: str | None = None
    proprio: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RobotAction:
    timestamp_s: float
    action_type: str
    values: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["values"] = list(self.values)
        return data


@dataclass(frozen=True)
class SteeringTrajectorySample:
    episode_id: str
    command: SteeringSkillCommand
    observations: list[SteeringObservation]
    actions: list[RobotAction]
    success: bool
    final_angle_error_rad: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "command": self.command.to_dict(),
            "observations": [item.to_dict() for item in self.observations],
            "actions": [item.to_dict() for item in self.actions],
            "success": self.success,
            "final_angle_error_rad": self.final_angle_error_rad,
            "metadata": self.metadata,
        }


DATASET_SCHEMA_VERSION = "robot_steering_skill_dataset_v0"


def dataset_record(sample: SteeringTrajectorySample) -> dict[str, Any]:
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        **sample.to_dict(),
    }

