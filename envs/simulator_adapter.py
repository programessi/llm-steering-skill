from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class SimObservation:
    rgb: np.ndarray
    depth_m: np.ndarray | None
    speed_mps: float
    steering_angle_rad: float
    timestamp: float


@dataclass
class TaskConfig:
    name: str
    horizon_steps: int = 240
    route_command: str = "keep_lane"
    initial_offset_m: float = 0.0
    road_curvature: float = 0.0
    has_front_vehicle: bool = False
    front_vehicle_distance_m: float = 40.0
    front_vehicle_speed_mps: float = 6.0
    distance_to_maneuver_m: float | None = None
    perception_noise: float = 0.0
    execution_noise: float = 0.0


class SimulatorAdapter(Protocol):
    task: TaskConfig

    def reset(self, task: TaskConfig) -> SimObservation:
        ...

    def step(self, steering_target_rad: float, target_speed_mps: float | None = None) -> SimObservation:
        ...

    def get_observation(self) -> SimObservation:
        ...

    def get_oracle_state_values(self) -> dict[str, float | bool | str | None]:
        ...

    def evaluate_after_skill(self, skill_name: str, target_angle: float | None) -> str:
        ...

    def task_finished(self) -> bool:
        ...

    def metrics(self) -> dict[str, float | int | bool | str]:
        ...

