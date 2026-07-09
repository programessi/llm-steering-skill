# Maintenance Notes

## What To Keep Under Version Control

Keep:

```text
README.md
GOAL.md
docs/
envs/
experiments/
feedback/
llm_policy/
perception/
robot_steering/
scripts/
skills/
```

Do not commit:

```text
.venv/
.venv310/
runs/
models/
third_party/
__pycache__/
```

## Current Canonical Smoke

```text
runs/model_perception_smoke/right_angle_left_deterministic_trilitenet_lane_yolo_marker_semantic_fixed_v3
```

This is the best current proof that the stage-1 chain works:

```text
CARLA front RGB
  -> TriLiteNet lane state + YOLO marker distance
  -> DrivingState
  -> generated policy code
  -> execute_primitive
  -> SteeringSkill
  -> feedback retry
  -> semantic success
```

## Safe Cleanup

Safe to delete and regenerate:

```text
__pycache__/
runs/debug_*
runs/tune_*
runs/discrete_debug_*
third_party/downloads/CARLA_0.9.15.segments*
third_party/downloads/*.bad.*
```

Keep unless intentionally reinstalling:

```text
third_party/carla
third_party/maniskill3
third_party/trilitenet
models/yolo11n.pt
models/trilitenet/*.pth
```

The `third_party/downloads/CARLA_0.9.15.tar.gz` archive is optional. It is not
needed at runtime once `third_party/carla` is installed, but keeping it avoids a
large redownload.
