from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.simulator_adapter import SimObservation
from perception.trilitenet_lane_adapter import TriLiteNetLanePerceptionAdapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=str(ROOT / "runs" / "model_perception_smoke" / "marker_frame.png"))
    parser.add_argument("--out", default=str(ROOT / "runs" / "model_perception_smoke" / "trilitenet_lane_offline_smoke"))
    parser.add_argument("--trilitenet-root", default=str(ROOT / "third_party" / "trilitenet"))
    parser.add_argument("--trilitenet-config", default="small")
    parser.add_argument("--trilitenet-weights", default=str(ROOT / "models" / "trilitenet" / "small.pth"))
    parser.add_argument("--trilitenet-device", default="cpu")
    parser.add_argument("--trilitenet-input-size", type=int, default=640)
    args = parser.parse_args()

    image_path = Path(args.image)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    adapter = TriLiteNetLanePerceptionAdapter(
        trilitenet_root=args.trilitenet_root,
        model_config=args.trilitenet_config,
        weights_path=args.trilitenet_weights,
        device=args.trilitenet_device,
        input_size=args.trilitenet_input_size,
    )
    da_mask, ll_mask = adapter._infer_masks(rgb)
    state = adapter.estimate(SimObservation(rgb=rgb, depth_m=None, speed_mps=0.0, steering_angle_rad=0.0, timestamp=0.0))

    overlay = bgr.copy()
    overlay[da_mask > 0] = (0.55 * overlay[da_mask > 0] + 0.45 * np.array([80, 180, 80])).astype(np.uint8)
    overlay[ll_mask > 0] = (0.35 * overlay[ll_mask > 0] + 0.65 * np.array([0, 0, 255])).astype(np.uint8)
    cv2.imwrite(str(out_dir / "trilitenet_overlay.png"), overlay)
    cv2.imwrite(str(out_dir / "lane_mask.png"), (ll_mask > 0).astype(np.uint8) * 255)
    cv2.imwrite(str(out_dir / "drivable_mask.png"), (da_mask > 0).astype(np.uint8) * 255)

    summary = {
        "image": str(image_path),
        "perception_adapter": "TriLiteNetLanePerceptionAdapter",
        "state_source": "front_rgb_trilitenet_lane",
        "lane_center_offset_m": state.lane_center_offset_m.value,
        "lane_center_offset_confidence": state.lane_center_offset_m.confidence,
        "heading_error_rad": state.heading_error_rad.value,
        "lane_curvature": state.lane_curvature.value,
        "drivable_area_confidence": state.drivable_area_confidence.value,
        "debug": adapter.last_debug.__dict__,
        "overlay": str(out_dir / "trilitenet_overlay.png"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
