from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Estimated:
    value: Any
    confidence: float
    timestamp: float

    def as_float(self, default: float = 0.0) -> float:
        if self.value is None:
            return default
        return float(self.value)


@dataclass
class DrivingState:
    speed_mps: Estimated
    steering_angle_rad: Estimated

    lane_center_offset_m: Estimated
    heading_error_rad: Estimated
    lane_curvature: Estimated
    drivable_area_confidence: Estimated

    front_vehicle_exists: Estimated
    front_vehicle_distance_m: Estimated
    front_vehicle_relative_speed_mps: Estimated

    route_command: Estimated
    distance_to_maneuver_m: Estimated

    perception_confidence: Estimated

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimated(value: Any, confidence: float, timestamp: float) -> Estimated:
    return Estimated(value=value, confidence=float(confidence), timestamp=float(timestamp))

