from __future__ import annotations

from envs.metadrive_adapter import MetaDriveAdapter
from envs.simulator_adapter import SimObservation
from perception.state_schema import DrivingState, estimated


class MetaDriveRenderedStateAdapter:
    """Adapter for MetaDrive rendered observations.

    It consumes `SimObservation.rgb` through the normal perception interface so
    policy code uses the same path as model-backed perception. For now, the
    structured lane fields are still taken from MetaDrive oracle state because
    Panda3D front RGB camera rendering is unavailable in the current headless
    environment. Replace this class with a model-backed adapter when an RGB
    camera frame is available.
    """

    def __init__(self, env: MetaDriveAdapter):
        self.env = env

    def estimate(self, obs: SimObservation) -> DrivingState:
        values = self.env.get_oracle_state_values()
        confidence = 1.0 if obs.rgb is not None and obs.rgb.size else 0.0
        return DrivingState(
            speed_mps=estimated(values["speed_mps"], 1.0, obs.timestamp),
            steering_angle_rad=estimated(values["steering_angle_rad"], 1.0, obs.timestamp),
            lane_center_offset_m=estimated(values["lane_center_offset_m"], confidence, obs.timestamp),
            heading_error_rad=estimated(values["heading_error_rad"], confidence, obs.timestamp),
            lane_curvature=estimated(values["lane_curvature"], confidence, obs.timestamp),
            drivable_area_confidence=estimated(1.0, confidence, obs.timestamp),
            front_vehicle_exists=estimated(False, 1.0, obs.timestamp),
            front_vehicle_distance_m=estimated(None, 0.0, obs.timestamp),
            front_vehicle_relative_speed_mps=estimated(None, 0.0, obs.timestamp),
            route_command=estimated("keep_lane", 1.0, obs.timestamp),
            distance_to_maneuver_m=estimated(None, 0.0, obs.timestamp),
            perception_confidence=estimated(confidence, confidence, obs.timestamp),
        )
