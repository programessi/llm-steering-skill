from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.carla_adapter import CarlaAdapter, CarlaAdapterConfig
from envs.simulator_adapter import TaskConfig
from llm_policy.code_generator import PolicyCodeGenerator
from llm_policy.policy_runtime import RestrictedPolicyRuntime
from skills.oracle_steering_skill import OracleSteeringSkill


def draw_dashboard(frame_rgb: np.ndarray, trace_row: dict, summary: dict, idx: int) -> np.ndarray:
    canvas = frame_rgb.copy()
    h, w = canvas.shape[:2]
    panel_w = min(360, max(280, int(w * 0.30)))
    x0 = w - panel_w
    cv2.rectangle(canvas, (x0, 0), (w, h), (20, 24, 28), -1)
    cv2.line(canvas, (x0, 0), (x0, h), (85, 95, 105), 1)

    def text(label: str, value: str, y: int, color=(232, 236, 240)) -> None:
        cv2.putText(canvas, label, (x0 + 18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 160, 168), 1)
        cv2.putText(canvas, value, (x0 + 18, y + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 2)

    text("Simulator", "CARLA front RGB", 34)
    text("State source", summary.get("state_source", "carla_oracle"), 92)
    text("Primitive", str(trace_row.get("steering_primitive") or "-"), 150)
    text("Speed", f"{float(trace_row['speed_mps']):.2f} m/s", 208)
    text("Lane offset", f"{float(trace_row['lane_center_offset_m']):+.3f} m", 266)
    text("Heading error", f"{float(trace_row['heading_error_rad']):+.3f} rad", 324)
    text("Steering target", f"{float(trace_row['steering_angle_rad']):+.3f} rad", 382)

    cx, cy = x0 + 88, min(h - 96, 476)
    cv2.circle(canvas, (cx, cy), 44, (80, 90, 98), 2)
    steer = float(trace_row["steering_angle_rad"])
    angle = -np.pi / 2 + np.clip(steer / 0.45, -1.0, 1.0) * np.pi * 0.7
    tip = (int(cx + 36 * np.cos(angle)), int(cy + 36 * np.sin(angle)))
    cv2.line(canvas, (cx, cy), tip, (70, 210, 255), 3)

    bx, by = x0 + 164, min(h - 124, 448)
    cv2.rectangle(canvas, (bx, by), (bx + 130, by + 18), (55, 62, 68), -1)
    cv2.line(canvas, (bx + 65, by - 5), (bx + 65, by + 23), (220, 225, 228), 1)
    offset = np.clip(float(trace_row["lane_center_offset_m"]) / 2.0, -1.0, 1.0)
    cv2.circle(canvas, (int(bx + 65 + offset * 60), by + 9), 8, (80, 255, 150), -1)
    cv2.putText(canvas, f"frame {idx:04d}", (18, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (235, 238, 240), 2)
    return canvas


def write_video(path: Path, frames_rgb: list[np.ndarray], trace: list[dict], summary: dict, fps: int) -> None:
    if not frames_rgb:
        return
    frames = [
        draw_dashboard(frame, row, summary, idx)
        for idx, (frame, row) in enumerate(zip(frames_rgb, trace[-len(frames_rgb) :]))
    ]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frames[0].shape[1], frames[0].shape[0]),
    )
    for frame in frames:
        writer.write(frame[:, :, ::-1])
    writer.release()
    cv2.imwrite(str(path.with_name("preview.png")), frames[min(len(frames) - 1, 8)][:, :, ::-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "runs" / "carla_rendered_demo"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default=None)
    parser.add_argument("--spawn-index", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=240)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    args = parser.parse_args()

    config = CarlaAdapterConfig(
        host=args.host,
        port=args.port,
        town=args.town,
        spawn_index=args.spawn_index,
        width=args.width,
        height=args.height,
        fixed_delta_seconds=1.0 / float(args.fps),
        timeout_s=args.timeout_s,
    )
    env = CarlaAdapter(config)
    try:
        env.reset(TaskConfig(name="carla_rendered_lane_keep", horizon_steps=args.horizon))
        skill = OracleSteeringSkill(env, dt=config.fixed_delta_seconds)
        code = PolicyCodeGenerator(use_codex=False).generate("CARLA front RGB lane keeping", mode="llm_feedback")
        runtime = RestrictedPolicyRuntime(env=env, steering_skill=skill, use_oracle_state=True)
        stats = runtime.run(code)
        summary = {
            **env.metrics(),
            "state_source": "carla_map_oracle",
            "camera_frame": "carla_front_rgb",
            "policy_mode": "llm_feedback",
            "valid_generated_code": stats.valid_code,
            "primitive_call_count": stats.primitive_call_count,
            "steering_primitive_counts": stats.steering_primitive_counts,
            "steering_primitive_sequence": stats.steering_primitive_sequence,
            "events": stats.events,
        }

        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        (out / "policy.py").write_text(code)
        if env.trace:
            with (out / "trace.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(env.trace[0].keys()))
                writer.writeheader()
                writer.writerows(env.trace)
        write_video(out / "front_rgb.mp4", env.camera_frames, env.trace, summary, args.fps)
        print(json.dumps(summary, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
