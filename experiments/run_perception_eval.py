from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.kinematic_simulator import KinematicDrivingSimulator, make_stage1_tasks
from perception.perception_adapter import FrontViewCVPerceptionAdapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "runs" / "perception_eval.json"))
    args = parser.parse_args()

    rows = []
    for task in make_stage1_tasks():
        env = KinematicDrivingSimulator()
        env.reset(task)
        def route_provider():
            values = env.get_oracle_state_values()
            return str(values["route_command"]), values["distance_to_maneuver_m"]

        perception = FrontViewCVPerceptionAdapter(route_provider=route_provider)
        for _ in range(min(task.horizon_steps, 120)):
            obs = env.get_observation()
            state = perception.estimate(obs)
            oracle = env.get_oracle_state_values()
            rows.append(
                {
                    "task": task.name,
                    "offset_abs_error": abs(
                        state.lane_center_offset_m.as_float() - float(oracle["lane_center_offset_m"])
                    ),
                    "heading_abs_error": abs(
                        state.heading_error_rad.as_float() - float(oracle["heading_error_rad"])
                    ),
                    "front_vehicle_distance_abs_error": (
                        abs(
                            state.front_vehicle_distance_m.as_float()
                            - float(oracle["front_vehicle_distance_m"])
                        )
                        if oracle["front_vehicle_distance_m"] is not None
                        and state.front_vehicle_distance_m.value is not None
                        else None
                    ),
                    "confidence": state.perception_confidence.as_float(),
                }
            )
            oracle = env.get_oracle_state_values()
            steering = (
                -0.22 * float(oracle["lane_center_offset_m"])
                -0.95 * float(oracle["heading_error_rad"])
                + 4.2 * float(oracle["lane_curvature"])
            )
            env.step(max(-0.28, min(0.28, steering)), 8.0)

    dist_errors = [r["front_vehicle_distance_abs_error"] for r in rows if r["front_vehicle_distance_abs_error"] is not None]
    summary = {
        "samples": len(rows),
        "mean_lane_center_offset_error_m": float(np.mean([r["offset_abs_error"] for r in rows])),
        "mean_heading_error_rad": float(np.mean([r["heading_abs_error"] for r in rows])),
        "mean_front_vehicle_distance_error_m": float(np.mean(dist_errors)) if dist_errors else None,
        "mean_confidence": float(np.mean([r["confidence"] for r in rows])),
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
