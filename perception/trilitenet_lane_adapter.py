from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from envs.simulator_adapter import SimObservation
from perception.perception_adapter import PerceptionAdapter
from perception.state_schema import DrivingState, estimated


@dataclass
class TriLiteNetLaneDebug:
    model_config: str
    weights_path: str
    lane_pixel_count: int
    drivable_pixel_count: int
    lane_confidence: float
    drivable_confidence: float
    bottom_lane_center_px: float | None
    mid_lane_center_px: float | None
    px_per_m: float
    source: str


class TriLiteNetLanePerceptionAdapter(PerceptionAdapter):
    """Estimate lane-relative posture from TriLiteNet lane/drivable masks.

    This adapter intentionally uses only the front RGB image plus ego speed and
    steering telemetry from the simulator observation. CARLA map values are not
    read here; oracle lane values should be logged separately for evaluation.
    """

    def __init__(
        self,
        trilitenet_root: str | Path = "third_party/trilitenet",
        model_config: str = "small",
        weights_path: str | Path = "models/trilitenet/small.pth",
        device: str = "cpu",
        input_size: int = 640,
        lane_width_m: float = 3.6,
        route_command: str = "keep_lane",
        convert_rgb_to_bgr: bool = True,
    ):
        self.trilitenet_root = Path(trilitenet_root)
        self.model_config = self._normalize_config(model_config)
        self.weights_path = Path(weights_path)
        self.device_name = device
        self.input_size = int(input_size)
        self.lane_width_m = float(lane_width_m)
        self.route_command = route_command
        self.convert_rgb_to_bgr = bool(convert_rgb_to_bgr)
        self.model = None
        self.torch = None
        self.last_debug = TriLiteNetLaneDebug(
            model_config=self.model_config,
            weights_path=str(self.weights_path),
            lane_pixel_count=0,
            drivable_pixel_count=0,
            lane_confidence=0.0,
            drivable_confidence=0.0,
            bottom_lane_center_px=None,
            mid_lane_center_px=None,
            px_per_m=64.0,
            source="not_loaded",
        )
        self._load_model()

    def estimate(self, obs: SimObservation) -> DrivingState:
        da_mask, ll_mask = self._infer_masks(obs.rgb)
        lane = self._estimate_lane_from_masks(ll_mask, da_mask)
        confidence = float(lane["confidence"])
        return DrivingState(
            speed_mps=estimated(obs.speed_mps, 1.0, obs.timestamp),
            steering_angle_rad=estimated(obs.steering_angle_rad, 1.0, obs.timestamp),
            lane_center_offset_m=estimated(lane["offset_m"], confidence, obs.timestamp),
            heading_error_rad=estimated(lane["heading_rad"], confidence, obs.timestamp),
            lane_curvature=estimated(lane["curvature"], confidence * 0.8, obs.timestamp),
            drivable_area_confidence=estimated(lane["drivable_confidence"], lane["drivable_confidence"], obs.timestamp),
            front_vehicle_exists=estimated(False, 0.0, obs.timestamp),
            front_vehicle_distance_m=estimated(None, 0.0, obs.timestamp),
            front_vehicle_relative_speed_mps=estimated(None, 0.0, obs.timestamp),
            route_command=estimated(self.route_command, 0.8, obs.timestamp),
            distance_to_maneuver_m=estimated(None, 0.0, obs.timestamp),
            perception_confidence=estimated(confidence, confidence, obs.timestamp),
        )

    def _load_model(self) -> None:
        if not self.trilitenet_root.exists():
            raise FileNotFoundError(f"TriLiteNet root does not exist: {self.trilitenet_root}")
        if not self.weights_path.exists():
            raise FileNotFoundError(f"TriLiteNet weights do not exist: {self.weights_path}")
        root = str(self.trilitenet_root.resolve())
        if root not in sys.path:
            sys.path.insert(0, root)

        import torch
        from lib.config import cfg
        from lib.models import get_net

        cfg.defrost()
        cfg.config = self.model_config
        cfg.freeze()
        model = get_net(cfg)
        checkpoint = torch.load(str(self.weights_path), map_location=self.device_name)
        model.load_state_dict(checkpoint, strict=True)
        model.to(self.device_name)
        model.eval()
        self.torch = torch
        self.model = model
        self.last_debug.source = "loaded"

    @staticmethod
    def _normalize_config(model_config: str) -> str:
        aliases = {
            "nano": "small",
            "tiny": "small",
        }
        return aliases.get(str(model_config), str(model_config))

    def _infer_masks(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.model is None or self.torch is None:
            raise RuntimeError("TriLiteNet model is not loaded.")
        h, w = rgb.shape[:2]
        image = cv2.resize(rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        if self.convert_rgb_to_bgr:
            image = image[:, :, ::-1]
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        tensor = self.torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device_name)
        with self.torch.no_grad():
            _, da_seg_out, ll_seg_out = self.model(tensor)
            da_mask = self.torch.argmax(da_seg_out, dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
            ll_mask = self.torch.argmax(ll_seg_out, dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
        da_mask = cv2.resize(da_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        ll_mask = cv2.resize(ll_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return da_mask, ll_mask

    def _estimate_lane_from_masks(self, ll_mask: np.ndarray, da_mask: np.ndarray) -> dict[str, float]:
        h, w = ll_mask.shape[:2]
        ll_mask = (ll_mask > 0).astype(np.uint8)
        da_mask = (da_mask > 0).astype(np.uint8)
        ll_mask[: int(h * 0.38), :] = 0
        da_mask[: int(h * 0.30), :] = 0

        bottom_y = int(h * 0.84)
        mid_y = int(h * 0.62)
        top_y = int(h * 0.48)
        bottom_center, bottom_width, c_bottom = self._lane_center_at_row(ll_mask, da_mask, bottom_y)
        mid_center, _, c_mid = self._lane_center_at_row(ll_mask, da_mask, mid_y)
        top_center, _, c_top = self._lane_center_at_row(ll_mask, da_mask, top_y)

        lane_pixels = int(ll_mask.sum())
        drivable_pixels = int(da_mask.sum())
        image_pixels = float(max(1, h * w))
        lane_density = min(1.0, lane_pixels / (image_pixels * 0.025))
        drivable_density = min(1.0, drivable_pixels / (image_pixels * 0.45))

        if bottom_center is None:
            bottom_center = self._drivable_center_at_row(da_mask, bottom_y)
        if mid_center is None:
            mid_center = self._drivable_center_at_row(da_mask, mid_y)
        if top_center is None:
            top_center = self._drivable_center_at_row(da_mask, top_y)
        if bottom_center is None:
            bottom_center = w / 2.0
        if mid_center is None:
            mid_center = bottom_center
        if top_center is None:
            top_center = mid_center

        px_per_m = 64.0
        if bottom_width is not None and bottom_width > 40.0:
            px_per_m = float(bottom_width / self.lane_width_m)
        offset_m = -(float(bottom_center) - w / 2.0) / max(px_per_m, 1.0)
        heading_rad = -((float(mid_center) - float(bottom_center)) / max(bottom_y - mid_y, 1)) * 0.42
        near_slope = (float(mid_center) - float(bottom_center)) / max(bottom_y - mid_y, 1)
        far_slope = (float(top_center) - float(mid_center)) / max(mid_y - top_y, 1)
        curvature = float((far_slope - near_slope) * 0.025)

        row_confidence = max(c_bottom, min(c_bottom, c_mid), min(c_mid, c_top))
        confidence = max(0.05, min(0.9, 0.65 * row_confidence + 0.25 * lane_density + 0.10 * drivable_density))
        drivable_confidence = max(0.05, min(0.9, drivable_density))
        self.last_debug = TriLiteNetLaneDebug(
            model_config=self.model_config,
            weights_path=str(self.weights_path),
            lane_pixel_count=lane_pixels,
            drivable_pixel_count=drivable_pixels,
            lane_confidence=float(confidence),
            drivable_confidence=float(drivable_confidence),
            bottom_lane_center_px=float(bottom_center),
            mid_lane_center_px=float(mid_center),
            px_per_m=float(px_per_m),
            source="trilitenet_segmentation",
        )
        return {
            "offset_m": float(offset_m),
            "heading_rad": float(heading_rad),
            "curvature": float(curvature),
            "confidence": float(confidence),
            "drivable_confidence": float(drivable_confidence),
        }

    def _lane_center_at_row(
        self,
        ll_mask: np.ndarray,
        da_mask: np.ndarray,
        y: int,
    ) -> tuple[float | None, float | None, float]:
        h, w = ll_mask.shape[:2]
        y0 = max(0, y - 6)
        y1 = min(h, y + 7)
        cols = np.where(ll_mask[y0:y1, :] > 0)[1]
        if cols.size >= 8:
            center_x = w / 2.0
            left = cols[cols < center_x]
            right = cols[cols >= center_x]
            left_x = float(np.max(left)) if left.size else None
            right_x = float(np.min(right)) if right.size else None
            if left_x is not None and right_x is not None and right_x - left_x > 25.0:
                return (left_x + right_x) / 2.0, right_x - left_x, min(1.0, cols.size / 80.0)
            if left_x is not None:
                drivable = self._drivable_center_at_row(da_mask, y)
                if drivable is not None:
                    return drivable, None, min(0.55, cols.size / 100.0)
            if right_x is not None:
                drivable = self._drivable_center_at_row(da_mask, y)
                if drivable is not None:
                    return drivable, None, min(0.55, cols.size / 100.0)
        drivable = self._drivable_center_at_row(da_mask, y)
        if drivable is not None:
            return drivable, None, 0.25
        return None, None, 0.0

    @staticmethod
    def _drivable_center_at_row(da_mask: np.ndarray, y: int) -> float | None:
        h, _ = da_mask.shape[:2]
        band = da_mask[max(0, y - 5) : min(h, y + 6), :]
        cols = np.where(band > 0)[1]
        if cols.size < 20:
            return None
        return float((np.percentile(cols, 10) + np.percentile(cols, 90)) / 2.0)
