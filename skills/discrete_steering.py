from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SteeringPrimitiveSpec:
    name: str
    angle_rad: float
    steer_duration_s: float
    hold_duration_s: float = 0.0
    return_duration_s: float = 0.0


SOFT_LEFT = SteeringPrimitiveSpec("soft_left", -0.08, 0.24)
MEDIUM_LEFT = SteeringPrimitiveSpec("medium_left", -0.15, 0.28)
HARD_LEFT = SteeringPrimitiveSpec("hard_left", -0.24, 0.34)

SOFT_RIGHT = SteeringPrimitiveSpec("soft_right", 0.08, 0.24)
MEDIUM_RIGHT = SteeringPrimitiveSpec("medium_right", 0.15, 0.28)
HARD_RIGHT = SteeringPrimitiveSpec("hard_right", 0.24, 0.34)

HOLD_CENTER = SteeringPrimitiveSpec("hold_center", 0.0, 0.25)
