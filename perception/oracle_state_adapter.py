from __future__ import annotations

from envs.simulator_adapter import SimulatorAdapter
from perception.state_schema import DrivingState, estimated


class OracleStateAdapter:
    def __init__(self, env: SimulatorAdapter):
        self.env = env

    def estimate_from_env(self) -> DrivingState:
        obs = self.env.get_observation()
        values = self.env.get_oracle_state_values()
        return DrivingState(
            speed_mps=estimated(values["speed_mps"], 1.0, obs.timestamp),
            steering_angle_rad=estimated(values["steering_angle_rad"], 1.0, obs.timestamp),
            lane_center_offset_m=estimated(values["lane_center_offset_m"], 1.0, obs.timestamp),
            heading_error_rad=estimated(values["heading_error_rad"], 1.0, obs.timestamp),
            lane_curvature=estimated(values["lane_curvature"], 1.0, obs.timestamp),
            drivable_area_confidence=estimated(values["drivable_area_confidence"], 1.0, obs.timestamp),
            front_vehicle_exists=estimated(values["front_vehicle_exists"], 1.0, obs.timestamp),
            front_vehicle_distance_m=estimated(values["front_vehicle_distance_m"], 1.0, obs.timestamp),
            front_vehicle_relative_speed_mps=estimated(values["front_vehicle_relative_speed_mps"], 1.0, obs.timestamp),
            route_command=estimated(values["route_command"], 1.0, obs.timestamp),
            distance_to_maneuver_m=estimated(values["distance_to_maneuver_m"], 1.0, obs.timestamp),
            perception_confidence=estimated(values["perception_confidence"], 1.0, obs.timestamp),
        )

