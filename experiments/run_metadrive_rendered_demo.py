from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.metadrive_adapter import MetaDriveAdapter
from envs.simulator_adapter import TaskConfig
from llm_policy.code_generator import PolicyCodeGenerator
from llm_policy.policy_runtime import RestrictedPolicyRuntime
from perception.metadrive_state_adapter import MetaDriveRenderedStateAdapter
from skills.oracle_steering_skill import OracleSteeringSkill


def draw_dashboard(frame: np.ndarray, trace_row: dict, summary: dict, idx: int) -> np.ndarray:
    canvas = frame.copy()
    h, w = canvas.shape[:2]
    panel_w = 340
    cv2.rectangle(canvas, (w - panel_w, 0), (w, h), (22, 25, 28), -1)
    cv2.line(canvas, (w - panel_w, 0), (w - panel_w, h), (90, 98, 104), 1)

    def text(label: str, value: str, y: int, color=(235, 238, 240)) -> None:
        cv2.putText(canvas, label, (w - panel_w + 18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150, 160, 168), 1)
        cv2.putText(canvas, value, (w - panel_w + 18, y + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2)

    text("Policy", summary.get("policy_mode", "llm_feedback"), 36)
    text("Task", summary.get("task", "metadrive_rendered"), 96)
    text("Speed", f"{float(trace_row['speed_mps']):.2f} m/s", 156)
    text("Lane offset", f"{float(trace_row['lane_center_offset_m']):+.3f} m", 216)
    text("Heading error", f"{float(trace_row['heading_error_rad']):+.3f} rad", 276)
    text("Steering", f"{float(trace_row['steering_angle_rad']):+.3f} rad", 336)

    cx, cy = w - panel_w + 92, 432
    cv2.circle(canvas, (cx, cy), 48, (85, 95, 102), 2)
    steer = float(trace_row["steering_angle_rad"])
    angle = -np.pi / 2 + np.clip(steer / 0.45, -1.0, 1.0) * np.pi * 0.7
    tip = (int(cx + 40 * np.cos(angle)), int(cy + 40 * np.sin(angle)))
    cv2.line(canvas, (cx, cy), tip, (65, 205, 255), 3)
    cv2.putText(canvas, "steering", (cx - 42, cy + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 170, 176), 1)

    bx, by = w - panel_w + 172, 402
    cv2.rectangle(canvas, (bx, by), (bx + 132, by + 18), (58, 64, 69), -1)
    cv2.line(canvas, (bx + 66, by - 5), (bx + 66, by + 23), (220, 220, 220), 1)
    offset = np.clip(float(trace_row["lane_center_offset_m"]) / 1.8, -1.0, 1.0)
    px = int(bx + 66 + offset * 60)
    cv2.circle(canvas, (px, by + 9), 8, (80, 255, 150), -1)
    cv2.putText(canvas, "lane offset", (bx + 20, by + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 170, 176), 1)

    cv2.putText(canvas, f"MetaDrive rendered frame {idx:03d}", (20, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 24, 28), 2)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "runs" / "metadrive_rendered_demo"))
    parser.add_argument("--horizon", type=int, default=160)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    args = parser.parse_args()

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

    env = MetaDriveAdapter(render_topdown=True, render_size=(args.width, args.height))
    try:
        env.reset(TaskConfig(name="metadrive_rendered_lane_keep", horizon_steps=args.horizon))
        skill = OracleSteeringSkill(env)
        perception = MetaDriveRenderedStateAdapter(env)
        code = PolicyCodeGenerator(use_codex=False).generate("MetaDrive rendered lane keeping", mode="llm_feedback")
        runtime = RestrictedPolicyRuntime(env=env, steering_skill=skill, perception=perception, use_oracle_state=False)
        stats = runtime.run(code)
        summary = {
            **env.metrics(),
            "state_source": "metadrive_rendered_frame_plus_lane_oracle",
            "policy_mode": "llm_feedback",
            "valid_generated_code": stats.valid_code,
            "primitive_call_count": stats.primitive_call_count,
            "events": stats.events,
            "render_source": "MetaDrive topdown renderer",
            "front_rgb_camera_status": "offscreen RGBCamera failed in current headless Panda3D/simplePBR environment",
        }

        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        (out / "policy.py").write_text(code)
        with (out / "trace.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(env.trace[0].keys()))
            writer.writeheader()
            writer.writerows(env.trace)

        rows = env.trace[-len(env.render_frames) :] if env.render_frames else []
        frames = [draw_dashboard(frame, row, summary, idx) for idx, (frame, row) in enumerate(zip(env.render_frames, rows))]
        if frames:
            video_path = out / "metadrive_rendered_demo.mp4"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                args.fps,
                (frames[0].shape[1], frames[0].shape[0]),
            )
            for frame in frames:
                writer.write(frame[..., ::-1])
            writer.release()
            cv2.imwrite(str(out / "preview.png"), frames[min(len(frames) - 1, 8)][..., ::-1])

        print(json.dumps(summary, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
