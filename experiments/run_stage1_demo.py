from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.kinematic_simulator import KinematicDrivingSimulator, make_stage1_tasks
from llm_policy.code_generator import PolicyCodeGenerator
from llm_policy.policy_runtime import RestrictedPolicyRuntime
from perception.perception_adapter import FrontViewCVPerceptionAdapter
from skills.oracle_steering_skill import OracleSteeringSkill


def run_one(task_name: str, state_source: str, out_dir: Path, make_video: bool, policy_mode: str = "llm_feedback") -> dict:
    task = next(task for task in make_stage1_tasks() if task.name == task_name)
    env = KinematicDrivingSimulator()
    env.reset(task)
    def route_provider():
        values = env.get_oracle_state_values()
        return str(values["route_command"]), values["distance_to_maneuver_m"]

    def curvature_provider():
        values = env.get_oracle_state_values()
        return float(values["lane_curvature"] or 0.0)

    perception = FrontViewCVPerceptionAdapter(
        noise_std=task.perception_noise,
        route_provider=route_provider,
        curvature_provider=curvature_provider,
    )
    skill = OracleSteeringSkill(env)
    code = PolicyCodeGenerator(use_codex=False).generate(task_description=task.name, mode=policy_mode)
    runtime = RestrictedPolicyRuntime(
        env=env,
        steering_skill=skill,
        perception=perception,
        use_oracle_state=(state_source == "oracle"),
    )

    frames = []
    stats = runtime.run(code)
    metrics = env.metrics()
    result = {
        **metrics,
        "state_source": state_source,
        "policy_mode": policy_mode,
        "valid_generated_code": stats.valid_code,
        "primitive_call_count": stats.primitive_call_count,
        "steering_primitive_counts": stats.steering_primitive_counts,
        "steering_primitive_sequence": stats.steering_primitive_sequence,
        "feedback_recovery_count": stats.feedback_recovery_count,
        "low_confidence_actions": stats.low_confidence_actions,
        "events": stats.events,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "policy.py").write_text(code)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2))
    with (out_dir / "trace.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(env.trace[0].keys()))
        writer.writeheader()
        writer.writerows(env.trace)

    if make_video:
        replay = KinematicDrivingSimulator()
        replay.reset(task)
        for row in env.trace[1:]:
            replay.lane_center_offset_m = float(row["lane_center_offset_m"])
            replay.heading_error_rad = float(row["heading_error_rad"])
            replay.steering_angle_rad = float(row["steering_angle_rad"])
            replay.speed_mps = float(row["speed_mps"])
            replay.time_s = float(row["time_s"])
            replay.step_count = int(row["step"])
            if row["front_vehicle_distance_m"] is not None:
                replay.front_vehicle_distance_m = float(row["front_vehicle_distance_m"])
            if row["distance_to_maneuver_m"] is not None:
                replay.distance_to_maneuver_m = float(row["distance_to_maneuver_m"])
            frame = replay.get_observation().rgb.copy()
            text = f"{task_name} offset={row['lane_center_offset_m']:.2f} steer={row['steering_angle_rad']:.2f}"
            cv2.putText(frame, text, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
            frames.append(frame)
        if frames:
            path = out_dir / "front_view.mp4"
            writer = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                20,
                (frames[0].shape[1], frames[0].shape[0]),
            )
            for frame in frames:
                writer.write(frame)
            writer.release()

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="lane_keeping")
    parser.add_argument("--state-source", choices=["perceived", "oracle"], default="perceived")
    parser.add_argument("--policy-mode", choices=["llm_feedback", "llm_no_feedback", "rule"], default="llm_feedback")
    parser.add_argument("--out", default=str(ROOT / "runs" / "stage1_demo"))
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()

    result = run_one(args.task, args.state_source, Path(args.out), args.video, args.policy_mode)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
