from __future__ import annotations

import cv2
import numpy as np

from collections.abc import Callable

from envs.simulator_adapter import SimObservation
from perception.state_schema import DrivingState, estimated


class PerceptionAdapter:
    def estimate(self, obs: SimObservation) -> DrivingState:
        raise NotImplementedError


class FrontViewCVPerceptionAdapter(PerceptionAdapter):
    """Simple front-view adapter for the local fallback simulator.

    This is deliberately replaceable. A TriLiteNet/TwinLiteNet/YOLO11 backend
    should implement the same estimate() contract and reuse the postprocess.
    """

    def __init__(
        self,
        lane_width_m: float = 3.6,
        rng_seed: int = 7,
        noise_std: float = 0.0,
        route_provider: Callable[[], tuple[str, float | None]] | None = None,
        curvature_provider: Callable[[], float] | None = None,
    ):
        self.lane_width_m = lane_width_m
        self.rng = np.random.default_rng(rng_seed)
        self.noise_std = noise_std
        self.route_provider = route_provider
        self.curvature_provider = curvature_provider
        self.last_front_distance: float | None = None
        self.last_timestamp: float | None = None

    def estimate(self, obs: SimObservation) -> DrivingState:
        rgb = obs.rgb
        h, w = rgb.shape[:2]
        lane = self._estimate_lane(rgb)
        vehicle_exists, vehicle_distance = self._estimate_front_vehicle(rgb, obs.depth_m)

        rel_speed = None
        if vehicle_exists and vehicle_distance is not None and self.last_front_distance is not None:
            dt = max(obs.timestamp - (self.last_timestamp or obs.timestamp), 1e-6)
            rel_speed = (vehicle_distance - self.last_front_distance) / dt
        if vehicle_exists:
            self.last_front_distance = vehicle_distance
            self.last_timestamp = obs.timestamp

        if self.route_provider is not None:
            route_command, distance_to_maneuver = self.route_provider()
        else:
            route_command = "keep_lane"
            distance_to_maneuver = None
        route_curvature = lane["curvature"]
        if self.curvature_provider is not None:
            route_curvature = self.curvature_provider()
        if route_command in ("turn_left", "turn_right") and distance_to_maneuver is not None:
            if -18.0 <= float(distance_to_maneuver) <= 12.0:
                route_curvature = -0.052 if route_command == "turn_left" else 0.052

        confidence = min(lane["confidence"], 0.95)
        if self.noise_std:
            lane["offset_m"] += float(self.rng.normal(0.0, self.noise_std))
            lane["heading_rad"] += float(self.rng.normal(0.0, self.noise_std * 0.08))
            confidence = max(0.2, confidence - self.noise_std)

        return DrivingState(
            speed_mps=estimated(obs.speed_mps, 1.0, obs.timestamp),
            steering_angle_rad=estimated(obs.steering_angle_rad, 1.0, obs.timestamp),
            lane_center_offset_m=estimated(lane["offset_m"], confidence, obs.timestamp),
            heading_error_rad=estimated(lane["heading_rad"], confidence, obs.timestamp),
            lane_curvature=estimated(route_curvature, confidence * 0.8, obs.timestamp),
            drivable_area_confidence=estimated(confidence, confidence, obs.timestamp),
            front_vehicle_exists=estimated(vehicle_exists, 0.85, obs.timestamp),
            front_vehicle_distance_m=estimated(vehicle_distance, 0.8 if vehicle_exists else 0.4, obs.timestamp),
            front_vehicle_relative_speed_mps=estimated(rel_speed, 0.65 if rel_speed is not None else 0.3, obs.timestamp),
            route_command=estimated(route_command, 0.5, obs.timestamp),
            distance_to_maneuver_m=estimated(distance_to_maneuver, 0.0, obs.timestamp),
            perception_confidence=estimated(confidence, confidence, obs.timestamp),
        )

    def _estimate_lane(self, rgb: np.ndarray) -> dict[str, float]:
        h, w = rgb.shape[:2]
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        mask[: int(h * 0.35), :] = 0

        def row_center(y: int) -> tuple[float | None, float]:
            band = mask[max(0, y - 4) : min(h, y + 5), :]
            cols = np.where(band > 0)[1]
            if cols.size < 20:
                return None, 0.0
            left_edge = float(np.percentile(cols, 15))
            right_edge = float(np.percentile(cols, 85))
            if right_edge - left_edge < 30:
                return None, 0.0
            return (left_edge + right_edge) / 2.0, min(1.0, cols.size / 200.0)

        bottom_y = int(h * 0.84)
        mid_y = int(h * 0.58)
        bottom_center, c1 = row_center(bottom_y)
        mid_center, c2 = row_center(mid_y)
        if bottom_center is None:
            return {"offset_m": 0.0, "heading_rad": 0.0, "curvature": 0.0, "confidence": 0.2}
        if mid_center is None:
            mid_center = bottom_center

        px_per_m_bottom = 64.0
        offset_m = -(bottom_center - w / 2.0) / px_per_m_bottom
        heading_rad = -((mid_center - bottom_center) / max(bottom_y - mid_y, 1)) * 0.42
        curvature = heading_rad * 0.025
        return {
            "offset_m": float(offset_m),
            "heading_rad": float(heading_rad),
            "curvature": float(curvature),
            "confidence": float(max(0.2, min(c1, c2))),
        }

    def _estimate_front_vehicle(
        self, rgb: np.ndarray, depth_m: np.ndarray | None
    ) -> tuple[bool, float | None]:
        red = rgb[:, :, 2].astype(np.int16)
        blue = rgb[:, :, 0].astype(np.int16)
        green = rgb[:, :, 1].astype(np.int16)
        mask = (red > 140) & (red > blue + 60) & (red > green + 60)
        ys, xs = np.where(mask)
        if xs.size < 30:
            return False, None
        if depth_m is not None:
            dist = float(np.median(depth_m[ys, xs]))
            return True, dist
        box_h = max(ys.max() - ys.min(), 1)
        return True, float(800.0 / box_h)
