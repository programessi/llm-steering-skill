from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot_steering.metrics import (
    aggregate_metrics,
    episode_hold_stability,
    episode_overshoot,
    episode_settling_time,
)
from robot_steering.schema import SteeringSkillCommand, dataset_record
from robot_steering.stub_skill import StubRobotSteeringSkill


COMMANDS = (
    SteeringSkillCommand("steer_to", 0.0, -0.30, 1.4, hold_s=0.4, max_speed_rad_s=0.45),
    SteeringSkillCommand("return_center", -0.30, 0.0, 1.1, hold_s=0.4, max_speed_rad_s=0.45),
    SteeringSkillCommand("steer_to", 0.0, 0.20, 1.0, hold_s=0.4, max_speed_rad_s=0.40),
    SteeringSkillCommand("steer_to", 0.20, -0.10, 1.2, hold_s=0.4, max_speed_rad_s=0.45),
    SteeringSkillCommand("hold", -0.10, -0.10, 0.8, hold_s=0.6, max_speed_rad_s=0.35),
)


def write_episode_csv(path: Path, sample) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp_s",
                "wheel_angle_rad",
                "target_angle_rad",
                "action_type",
                "action_values",
            ],
        )
        writer.writeheader()
        for obs, action in zip(sample.observations, sample.actions):
            writer.writerow(
                {
                    "timestamp_s": f"{obs.timestamp_s:.4f}",
                    "wheel_angle_rad": f"{obs.wheel_angle_rad:.6f}",
                    "target_angle_rad": f"{obs.target_angle_rad:.6f}",
                    "action_type": action.action_type,
                    "action_values": " ".join(f"{value:.6f}" for value in action.values),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "runs" / "robot_steering_skill_bench"))
    parser.add_argument("--dt-s", type=float, default=0.05)
    parser.add_argument("--response-rate", type=float, default=5.0)
    parser.add_argument("--angle-tolerance-rad", type=float, default=0.035)
    args = parser.parse_args()

    out = Path(args.out)
    episodes_dir = out / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    skill = StubRobotSteeringSkill(
        dt_s=args.dt_s,
        response_rate=args.response_rate,
        angle_tolerance_rad=args.angle_tolerance_rad,
    )
    samples = []
    rows = []
    for idx, command in enumerate(COMMANDS, start=1):
        sample = skill.execute(command)
        samples.append(sample)
        episode_dir = episodes_dir / sample.episode_id
        episode_dir.mkdir(parents=True, exist_ok=True)
        (episode_dir / "sample.json").write_text(json.dumps(dataset_record(sample), indent=2))
        write_episode_csv(episode_dir / "trajectory.csv", sample)
        rows.append(
            {
                "episode_id": sample.episode_id,
                "skill_name": command.skill_name,
                "current_angle_rad": command.current_angle_rad,
                "target_angle_rad": command.target_angle_rad,
                "duration_s": command.duration_s,
                "hold_s": command.hold_s,
                "success": int(sample.success),
                "final_angle_error_rad": f"{sample.final_angle_error_rad:.6f}",
                "overshoot_rad": f"{episode_overshoot(sample):.6f}",
                "settling_time_s": f"{episode_settling_time(sample, args.angle_tolerance_rad):.4f}",
                "hold_stability_rad": f"{episode_hold_stability(sample):.6f}",
            }
        )
    metrics = aggregate_metrics(samples, tolerance_rad=args.angle_tolerance_rad)
    with (out / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "bench": "robot_steering_skill_bench",
        "skill_impl": "StubRobotSteeringSkill",
        "dataset_schema_version": "robot_steering_skill_dataset_v0",
        "angle_tolerance_rad": args.angle_tolerance_rad,
        "command_count": len(COMMANDS),
        "metrics": metrics.to_dict(),
        "results_csv": str(out / "results.csv"),
        "episodes_dir": str(episodes_dir),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

