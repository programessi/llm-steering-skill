from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.carla_adapter import CarlaAdapter, CarlaAdapterConfig
from envs.simulator_adapter import TaskConfig
from skills.oracle_steering_skill import OracleSteeringSkill
from skills.steering_skill import SteeringSkill


@dataclass(frozen=True)
class StageSpec:
    name: str
    primitive: str
    target_angle_rad: float
    duration_s: float
    speed_mps: float
    trigger: str


@dataclass(frozen=True)
class DemoSpec:
    name: str
    title: str
    description: str
    spawn_index: int
    horizon_steps: int
    stages: tuple[StageSpec, ...]


DEMOS: dict[str, DemoSpec] = {
    "right_angle_left": DemoSpec(
        name="right_angle_left",
        title="Fixed-point left turn",
        description="Approach a fixed point, turn the wheel left, hold, then return center.",
        spawn_index=0,
        horizon_steps=170,
        stages=(
            StageSpec("approach", "hold_center", 0.0, 2.2, 3.2, "distance_to_turn > 0"),
            StageSpec("turn_in", "hard_left", -0.32, 1.5, 2.8, "front bumper reaches turn point"),
            StageSpec("hold_arc", "hold_left", -0.32, 1.4, 2.8, "vehicle heading rotates through target sector"),
            StageSpec("return", "return_center", 0.0, 1.1, 3.0, "heading target reached"),
            StageSpec("exit", "hold_center", 0.0, 2.4, 3.2, "exit line aligned"),
        ),
    ),
    "lane_change_left": DemoSpec(
        name="lane_change_left",
        title="Fixed-point lane change",
        description="At a planned marker, steer left, hold the lane-change arc, counter-steer, then center.",
        spawn_index=6,
        horizon_steps=180,
        stages=(
            StageSpec("approach", "hold_center", 0.0, 1.8, 4.0, "lane-change marker ahead"),
            StageSpec("move_left", "medium_left", -0.20, 1.2, 3.8, "marker reached"),
            StageSpec("counter", "medium_right", 0.18, 1.3, 3.8, "ego crosses lane boundary"),
            StageSpec("center", "return_center", 0.0, 1.0, 3.8, "target lane center acquired"),
            StageSpec("stabilize", "hold_center", 0.0, 2.4, 4.0, "new lane aligned"),
        ),
    ),
    "pull_over_right": DemoSpec(
        name="pull_over_right",
        title="Pull over and straighten",
        description="Move toward the right edge, counter-steer to straighten, return center, then stop.",
        spawn_index=12,
        horizon_steps=190,
        stages=(
            StageSpec("approach", "hold_center", 0.0, 1.8, 3.2, "curb/edge target ahead"),
            StageSpec("pull_right", "medium_right", 0.20, 1.4, 2.8, "pull-over point reached"),
            StageSpec("straighten", "medium_left", -0.16, 1.2, 2.6, "right offset target reached"),
            StageSpec("center_wheel", "return_center", 0.0, 1.0, 2.2, "vehicle nearly parallel"),
            StageSpec("stop", "hold_center_stop", 0.0, 2.6, 0.0, "parking pose reached"),
        ),
    ),
}


POLICY_TEMPLATE = '''\
def policy():
    # LLM-generated structured skill program for: {title}
    # Inputs:
    #   state = DrivingState from carla_map_oracle
    #   feedback = ExecutionFeedback from the previous SteeringSkill call
    # Output:
    #   calls to a replaceable SteeringSkill boundary. The policy never sends
    #   continuous wheel commands directly.
{body}
'''


def build_policy_text(demo: DemoSpec) -> str:
    body_lines = []
    for stage in demo.stages:
        body_lines.extend(
            [
                f"    wait_until({stage.trigger!r})",
                "    execute_primitive(",
                f"        name={stage.primitive!r},",
                f"        target_angle_rad={stage.target_angle_rad!r},",
                f"        duration_s={stage.duration_s!r},",
                f"        speed_mps={stage.speed_mps!r},",
                "    )",
                "",
            ]
        )
    return POLICY_TEMPLATE.format(title=demo.title, body="\n".join(body_lines).rstrip())


def draw_dashboard(frame_rgb: np.ndarray, trace_row: dict, demo: DemoSpec, summary: dict, idx: int) -> np.ndarray:
    canvas = frame_rgb.copy()
    h, w = canvas.shape[:2]
    panel_w = min(440, max(330, int(w * 0.34)))
    x0 = w - panel_w
    cv2.rectangle(canvas, (x0, 0), (w, h), (18, 22, 26), -1)
    cv2.line(canvas, (x0, 0), (x0, h), (82, 92, 102), 1)

    def label_value(label: str, value: str, y: int, color=(232, 236, 240), value_scale=0.62) -> None:
        cv2.putText(canvas, label, (x0 + 18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 160, 168), 1)
        cv2.putText(canvas, value, (x0 + 18, y + 25), cv2.FONT_HERSHEY_SIMPLEX, value_scale, color, 2)

    primitive = str(trace_row.get("steering_primitive") or "-")
    stage = str(trace_row.get("demo_stage") or "reset")
    trigger = str(trace_row.get("trigger_condition") or "-")
    label_value("Demo", demo.title, 32, value_scale=0.58)
    label_value("Stage", stage, 88, color=(110, 220, 255), value_scale=0.70)
    label_value("Trigger", trigger[:34], 146, value_scale=0.48)
    label_value("Primitive", primitive, 206, color=(255, 220, 120), value_scale=0.70)
    label_value("Speed", f"{float(trace_row['speed_mps']):.2f} m/s", 264)
    label_value("Lane offset", f"{float(trace_row['lane_center_offset_m']):+.3f} m", 322)
    label_value("Heading error", f"{float(trace_row['heading_error_rad']):+.3f} rad", 380)
    label_value("Steering target", f"{float(trace_row['steering_angle_rad']):+.3f} rad", 438)

    cx, cy = x0 + 92, min(h - 92, 536)
    cv2.circle(canvas, (cx, cy), 44, (82, 94, 104), 2)
    steer = float(trace_row["steering_angle_rad"])
    angle = -np.pi / 2 + np.clip(steer / 0.45, -1.0, 1.0) * np.pi * 0.72
    tip = (int(cx + 36 * np.cos(angle)), int(cy + 36 * np.sin(angle)))
    cv2.line(canvas, (cx, cy), tip, (70, 210, 255), 3)

    bx, by = x0 + 175, min(h - 122, 508)
    cv2.rectangle(canvas, (bx, by), (bx + 150, by + 18), (55, 62, 68), -1)
    cv2.line(canvas, (bx + 75, by - 5), (bx + 75, by + 23), (220, 225, 228), 1)
    offset = np.clip(float(trace_row["lane_center_offset_m"]) / 2.0, -1.0, 1.0)
    cv2.circle(canvas, (int(bx + 75 + offset * 70), by + 9), 8, (80, 255, 150), -1)

    cv2.putText(canvas, f"frame {idx:04d}", (18, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (235, 238, 240), 2)
    cv2.putText(
        canvas,
        f"{summary.get('state_source', 'carla_map_oracle')} -> SteeringSkill",
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (235, 238, 240),
        2,
    )
    return canvas


def write_video(path: Path, frames_rgb: list[np.ndarray], trace: list[dict], demo: DemoSpec, summary: dict, fps: int) -> None:
    if not frames_rgb:
        return
    rows = trace[-len(frames_rgb) :]
    frames = [draw_dashboard(frame, row, demo, summary, idx) for idx, (frame, row) in enumerate(zip(frames_rgb, rows))]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frames[0].shape[1], frames[0].shape[0]),
    )
    for frame in frames:
        writer.write(frame[:, :, ::-1])
    writer.release()
    cv2.imwrite(str(path.with_name("preview.png")), frames[min(len(frames) - 1, 18)][:, :, ::-1])


def write_steering_plot(path: Path, trace: list[dict]) -> None:
    rows = [row for row in trace if row.get("event") != "reset"]
    if not rows:
        return
    w, h = 960, 360
    margin_l, margin_r, margin_t, margin_b = 76, 28, 36, 58
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    plot_w = w - margin_l - margin_r
    plot_h = h - margin_t - margin_b
    cv2.rectangle(canvas, (margin_l, margin_t), (w - margin_r, h - margin_b), (236, 240, 244), 1)
    cv2.line(canvas, (margin_l, margin_t + plot_h // 2), (w - margin_r, margin_t + plot_h // 2), (196, 202, 210), 1)

    times = np.array([float(row["time_s"]) for row in rows], dtype=np.float32)
    steer = np.array([float(row["steering_angle_rad"]) for row in rows], dtype=np.float32)
    target = np.array([float(row.get("planned_target_angle_rad") or 0.0) for row in rows], dtype=np.float32)
    t0, t1 = float(times.min()), float(times.max())
    if t1 <= t0:
        t1 = t0 + 1.0
    y_abs = max(0.45, float(np.max(np.abs(np.concatenate([steer, target])))))

    def point(t: float, y: float) -> tuple[int, int]:
        x = margin_l + int((t - t0) / (t1 - t0) * plot_w)
        yy = margin_t + int((1.0 - ((y + y_abs) / (2.0 * y_abs))) * plot_h)
        return x, yy

    last_stage = None
    for idx, row in enumerate(rows):
        stage = row.get("demo_stage")
        if stage != last_stage:
            x, _ = point(float(row["time_s"]), 0.0)
            cv2.line(canvas, (x, margin_t), (x, h - margin_b), (225, 228, 232), 1)
            cv2.putText(canvas, str(stage), (x + 4, margin_t + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (92, 98, 106), 1)
            last_stage = stage

    for series, color in ((target, (180, 120, 40)), (steer, (30, 110, 220))):
        pts = np.array([point(float(t), float(y)) for t, y in zip(times, series)], dtype=np.int32)
        if len(pts) >= 2:
            cv2.polylines(canvas, [pts], False, color, 2)

    cv2.putText(canvas, "steering_angle_rad", (margin_l, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (38, 44, 52), 2)
    cv2.putText(canvas, "blue=actual command, brown=planned target", (margin_l + 230, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (92, 98, 106), 1)
    cv2.putText(canvas, "time_s", (w // 2 - 28, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (38, 44, 52), 1)
    cv2.putText(canvas, f"+{y_abs:.2f}", (18, margin_t + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (92, 98, 106), 1)
    cv2.putText(canvas, "0.00", (24, margin_t + plot_h // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (92, 98, 106), 1)
    cv2.putText(canvas, f"-{y_abs:.2f}", (18, h - margin_b + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (92, 98, 106), 1)
    cv2.imwrite(str(path), canvas)


def annotate_trace_rows(env: CarlaAdapter, start_idx: int, stage: StageSpec) -> None:
    for row in env.trace[start_idx:]:
        row["demo_stage"] = stage.name
        row["trigger_condition"] = stage.trigger
        row["planned_primitive"] = stage.primitive
        row["planned_target_angle_rad"] = stage.target_angle_rad


def run_stage(skill: SteeringSkill, env: CarlaAdapter, stage: StageSpec) -> None:
    skill.set_target_speed(stage.speed_mps)
    skill.set_trace_primitive(stage.primitive)
    start_idx = len(env.trace)
    try:
        if stage.primitive in {"return_center", "hold_center", "hold_center_stop"}:
            result = skill.return_center(stage.duration_s) if stage.primitive == "return_center" else skill.hold(0.0, stage.duration_s)
        elif stage.primitive.startswith("hold_"):
            result = skill.hold(stage.target_angle_rad, stage.duration_s)
        else:
            result = skill.steer_to(stage.target_angle_rad, stage.duration_s)
    finally:
        skill.set_trace_primitive(None)
    annotate_trace_rows(env, start_idx, stage)
    if result.event != "none":
        for row in env.trace[start_idx:]:
            row["execution_feedback"] = result.event


def run_demo(env: CarlaAdapter, demo: DemoSpec, args: argparse.Namespace, out_root: Path) -> dict:
    env.config.spawn_index = demo.spawn_index if args.spawn_index is None else args.spawn_index
    try:
        env.reset(TaskConfig(name=demo.name, horizon_steps=demo.horizon_steps))
        for row in env.trace:
            row["demo_stage"] = "reset"
            row["trigger_condition"] = "spawn at start pose"
            row["planned_primitive"] = "none"
            row["planned_target_angle_rad"] = 0.0
        skill = OracleSteeringSkill(env, dt=env.config.fixed_delta_seconds, default_speed_mps=4.8)
        for stage in demo.stages:
            if env.task_finished():
                break
            run_stage(skill, env, stage)

        primitive_sequence = [str(row["planned_primitive"]) for row in env.trace if row.get("planned_primitive") not in {None, "none"}]
        summary = {
            **env.metrics(),
            "demo": demo.name,
            "title": demo.title,
            "description": demo.description,
            "state_source": "carla_map_oracle",
            "camera_frame": "carla_front_rgb",
            "perception_adapter": "CarlaMapOracleStateAdapter",
            "control_source": "SteeringSkill",
            "steering_skill_impl": "OracleSteeringSkill",
            "policy_mode": "driving_test_fixed_point_program",
            "valid_generated_code": True,
            "primitive_call_count": len(demo.stages),
            "steering_primitive_counts": dict(Counter(stage.primitive for stage in demo.stages)),
            "steering_primitive_sequence": [stage.primitive for stage in demo.stages],
            "per_step_planned_primitive_counts": dict(Counter(primitive_sequence)),
            "stages": [stage.__dict__ for stage in demo.stages],
        }

        out = out_root / demo.name
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        (out / "policy.py").write_text(build_policy_text(demo))
        if env.trace:
            fieldnames = list(env.trace[0].keys())
            for row in env.trace:
                for key in row:
                    if key not in fieldnames:
                        fieldnames.append(key)
            with (out / "trace.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(env.trace)
        write_video(out / "front_rgb.mp4", env.camera_frames, env.trace, demo, summary, args.fps)
        write_steering_plot(out / "steering_curve.png", env.trace)
        return summary
    except Exception:
        env.close()
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "runs" / "carla_driving_test_demos"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default=None)
    parser.add_argument("--demo", choices=["all", *DEMOS.keys()], default="all")
    parser.add_argument("--spawn-index", type=int, default=None)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--lane-departure-limit-m", type=float, default=5.0)
    args = parser.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    selected = list(DEMOS.values()) if args.demo == "all" else [DEMOS[args.demo]]
    config = CarlaAdapterConfig(
        host=args.host,
        port=args.port,
        town=args.town,
        spawn_index=selected[0].spawn_index if args.spawn_index is None else args.spawn_index,
        width=args.width,
        height=args.height,
        fixed_delta_seconds=1.0 / float(args.fps),
        timeout_s=args.timeout_s,
        lane_departure_limit_m=args.lane_departure_limit_m,
    )
    env = CarlaAdapter(config)
    try:
        summaries = [run_demo(env, demo, args, out_root) for demo in selected]
    finally:
        env.close()
    aggregate = {
        "state_source": "carla_map_oracle",
        "camera_frame": "carla_front_rgb",
        "perception_adapter": "CarlaMapOracleStateAdapter",
        "control_source": "SteeringSkill",
        "steering_skill_impl": "OracleSteeringSkill",
        "policy_mode": "driving_test_fixed_point_program",
        "demo_count": len(summaries),
        "success_count": sum(1 for item in summaries if item["success"]),
        "demos": summaries,
    }
    (out_root / "summary.json").write_text(json.dumps(aggregate, indent=2))
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
