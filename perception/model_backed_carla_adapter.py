from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from envs.simulator_adapter import SimObservation
from perception.perception_adapter import FrontViewCVPerceptionAdapter, PerceptionAdapter
from perception.state_schema import DrivingState, estimated


@dataclass
class ModelPerceptionDebug:
    marker_detected: bool
    marker_class_name: str | None
    marker_confidence: float
    marker_bbox_xyxy: tuple[float, float, float, float] | None
    marker_distance_m: float | None
    marker_distance_source: str


class ModelBackedCarlaPerceptionAdapter:
    """Front-RGB DrivingState adapter with a model-backed maneuver marker.

    v0 replaces the LLM's key trigger field, distance_to_maneuver_m, with a
    YOLO-detected visual marker distance. Lane fields come from image geometry
    rather than CARLA map state; oracle values are intentionally not used here.
    """

    def __init__(
        self,
        model_path: str | Path = "models/yolo11n.pt",
        marker_class: str = "car",
        marker_real_height_m: float = 1.50,
        camera_fov_deg: float = 90.0,
        confidence_threshold: float = 0.25,
        route_command: str = "turn_left",
        lane_adapter: PerceptionAdapter | None = None,
    ):
        from ultralytics import YOLO

        self.model_path = Path(model_path)
        self.model = YOLO(str(self.model_path))
        self.marker_class = marker_class
        self.marker_real_height_m = float(marker_real_height_m)
        self.camera_fov_deg = float(camera_fov_deg)
        self.confidence_threshold = float(confidence_threshold)
        self.route_command = route_command
        self.lane_adapter = lane_adapter or FrontViewCVPerceptionAdapter()
        self.last_debug = ModelPerceptionDebug(False, None, 0.0, None, None, "not_detected")
        self.last_marker_distance_m: float | None = None
        self.last_marker_timestamp: float | None = None

    def estimate(self, obs: SimObservation) -> DrivingState:
        lane_state = self.lane_adapter.estimate(obs)
        marker = self._estimate_marker_distance(obs.rgb)
        marker = self._stabilize_marker_distance(marker, obs)
        perception_confidence = max(
            0.05,
            min(
                0.95,
                float(marker.marker_confidence) if marker.marker_detected else 0.05,
            ),
        )
        return DrivingState(
            speed_mps=estimated(obs.speed_mps, 1.0, obs.timestamp),
            steering_angle_rad=estimated(obs.steering_angle_rad, 1.0, obs.timestamp),
            lane_center_offset_m=lane_state.lane_center_offset_m,
            heading_error_rad=lane_state.heading_error_rad,
            lane_curvature=lane_state.lane_curvature,
            drivable_area_confidence=lane_state.drivable_area_confidence,
            front_vehicle_exists=estimated(marker.marker_detected, perception_confidence, obs.timestamp),
            front_vehicle_distance_m=estimated(marker.marker_distance_m, perception_confidence, obs.timestamp),
            front_vehicle_relative_speed_mps=estimated(None, 0.0, obs.timestamp),
            route_command=estimated(self.route_command, 1.0, obs.timestamp),
            distance_to_maneuver_m=estimated(marker.marker_distance_m, perception_confidence, obs.timestamp),
            perception_confidence=estimated(perception_confidence, perception_confidence, obs.timestamp),
        )

    def _estimate_marker_distance(self, rgb: np.ndarray) -> ModelPerceptionDebug:
        results = self.model.predict(rgb, verbose=False, conf=self.confidence_threshold)
        result = results[0]
        names = result.names
        best = None
        for box in result.boxes:
            cls_id = int(box.cls.item())
            class_name = str(names.get(cls_id, cls_id))
            if class_name != self.marker_class:
                continue
            conf = float(box.conf.item())
            xyxy = tuple(float(v) for v in box.xyxy[0].detach().cpu().numpy().tolist())
            x1, y1, x2, y2 = xyxy
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if best is None or area > best[0]:
                best = (area, conf, class_name, xyxy)
        if best is None:
            self.last_debug = ModelPerceptionDebug(False, None, 0.0, None, None, "not_detected")
            return self.last_debug

        _, conf, class_name, xyxy = best
        _, y1, _, y2 = xyxy
        bbox_h = max(1.0, y2 - y1)
        image_h, image_w = rgb.shape[:2]
        focal_px = (image_w / 2.0) / math.tan(math.radians(self.camera_fov_deg) / 2.0)
        distance = (self.marker_real_height_m * focal_px) / bbox_h
        self.last_debug = ModelPerceptionDebug(
            marker_detected=True,
            marker_class_name=class_name,
            marker_confidence=conf,
            marker_bbox_xyxy=xyxy,
            marker_distance_m=float(distance),
            marker_distance_source="yolo_bbox_pinhole",
        )
        return self.last_debug

    def _stabilize_marker_distance(
        self,
        marker: ModelPerceptionDebug,
        obs: SimObservation,
    ) -> ModelPerceptionDebug:
        predicted = None
        if self.last_marker_distance_m is not None and self.last_marker_timestamp is not None:
            dt = max(0.0, obs.timestamp - self.last_marker_timestamp)
            predicted = max(0.0, self.last_marker_distance_m - max(0.0, obs.speed_mps) * dt)

        use_prediction = False
        if predicted is not None:
            if not marker.marker_detected:
                use_prediction = True
            elif marker.marker_distance_m is not None:
                jump = marker.marker_distance_m - self.last_marker_distance_m
                use_prediction = self.last_marker_distance_m < 6.0 and jump > 1.5

        if use_prediction and predicted is not None:
            marker = ModelPerceptionDebug(
                marker_detected=True,
                marker_class_name=marker.marker_class_name or self.marker_class,
                marker_confidence=min(max(marker.marker_confidence, 0.35), 0.55),
                marker_bbox_xyxy=marker.marker_bbox_xyxy,
                marker_distance_m=predicted,
                marker_distance_source="yolo_temporal_odometry",
            )

        if marker.marker_detected and marker.marker_distance_m is not None:
            self.last_marker_distance_m = float(marker.marker_distance_m)
            self.last_marker_timestamp = obs.timestamp
        return marker
