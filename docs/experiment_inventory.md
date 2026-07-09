# Experiment Inventory

This file keeps Stage-1 outputs organized as the project grows.

## Canonical Outputs

Use these directories for current results and reporting:

```text
runs/stage1_benchmark_report
runs/carla_stage1_auto_feedback_repeats_llm3
runs/auto_feedback_tasks/lane_change_left/repeats_llm3
```

Start from:

```text
runs/stage1_benchmark_report/README.md
```

Each repeat directory contains:

```text
summary.json
trials.csv
trial_XXX/summary.json
trial_XXX/attempt_01/policy.py
trial_XXX/attempt_01/trace.csv
trial_XXX/attempt_01/front_rgb.mp4
trial_XXX/attempt_02/policy.py
trial_XXX/attempt_02/trace.csv
trial_XXX/attempt_02/front_rgb.mp4
```

## Current Task Results

```text
right_angle_left:
  path=runs/carla_stage1_auto_feedback_repeats_llm3
  trials=3
  attempt1_success_rate=0.0
  final_success_rate=1.0
  attempt1_feedback_counts={"turn_too_early": 3}
  final_feedback_counts={"none": 3}

lane_change_left:
  path=runs/auto_feedback_tasks/lane_change_left/repeats_llm3
  trials=3
  attempt1_success_rate=0.0
  final_success_rate=1.0
  attempt1_feedback_counts={"turn_too_early": 3}
  final_feedback_counts={"none": 3}
```

## Stable Code Paths

```text
feedback/trace_feedback_adapter.py
experiments/run_carla_stage1_auto_feedback_repeats.py
scripts/run_carla_stage1_auto_feedback_repeats.sh
experiments/run_carla_stage1_conditional_policy.py
llm_policy/prompt_builder.py
llm_policy/llm_code_generator.py
skills/simulated_robot_steering_skill.py
robot_steering/stub_skill.py
robot_steering/valve_skill.py
robot_steering/backends.py
```

## Robot Skill Adapter Smoke

The oracle steering implementation can now be replaced at runtime:

```bash
scripts/run_carla_stage1_auto_feedback_repeats.sh \
  --task right_angle_left \
  --trials 1 \
  --policy-generator llm \
  --steering-skill simulated_robot \
  --out runs/robot_skill_adapter_smoke/right_angle_left_llm_smooth
```

Latest result:

```text
path=runs/robot_skill_adapter_smoke/right_angle_left_llm_smooth
policy_generator=llm
steering_skill_impl=SimulatedRobotSteeringSkill
robot_skill_impl=StubRobotSteeringSkill
attempt1_feedback_counts={"turn_too_early": 1}
final_feedback_counts={"none": 1}
final_success_rate=1.0
final_max_lane_center_offset_m=1.3690
```

Important trace fields:

```text
steering_skill_impl
robot_skill_impl
robot_steering_episode_id
robot_command_target_angle_rad
robot_observed_wheel_angle_rad
robot_action_delta_angle_rad
```

## Valve Backend Smoke

Current first-step valve bridge:

```text
path=runs/valve_backend_smoke/right_angle_left_deterministic_kinematic_valve
policy_generator=deterministic
steering_skill_impl=SimulatedRobotSteeringSkill
robot_skill_backend=kinematic_valve
robot_skill_impl=KinematicValveSteeringSkill
valve_joint_qpos_rad == steering_angle_rad
```

The LLM retry run also completed code generation and CARLA execution:

```text
path=runs/valve_backend_smoke/right_angle_left_llm_kinematic_valve_retry
policy_generator=llm
robot_skill_backend=kinematic_valve
valid_code_rate=1.0
attempt1_feedback_counts={"turn_too_early": 1}
final_feedback_counts={"lane_departure_risk": 1}
final_max_lane_center_offset_m=1.9504
```

The LLM run proves the control path, but its final policy did not pass the
current right-angle semantic threshold of 1.8 m. This is a policy tuning issue,
not a valve bridge issue: trace shows zero difference between
`valve_joint_qpos_rad` and CARLA `steering_angle_rad`.

ManiSkill3 source is now managed under:

```text
third_party/maniskill3
```

See:

```text
docs/maniskill_valve_backend.md
```

Installed ManiSkill backend smoke:

```text
path=runs/valve_backend_smoke/right_angle_left_deterministic_maniskill_valve
policy_generator=deterministic
robot_skill_backend=maniskill_valve
robot_skill_impl=ManiSkillValveSteeringSkill
trace_rows_with_valve=183
max_abs_diff_valve_vs_steer=0.0
```

## Model Perception Smoke

First non-oracle LLM state path:

```text
perception/model_backed_carla_adapter.py
models/yolo11n.pt
```

Run:

```bash
scripts/run_carla_stage1_auto_feedback_repeats.sh \
  --task right_angle_left \
  --trials 1 \
  --policy-generator deterministic \
  --steering-skill oracle \
  --perception-source model_marker \
  --visual-marker-distance-m 8.0 \
  --visual-marker-lateral-offset-m 2.2 \
  --visual-marker-real-height-m 2.0 \
  --out runs/model_perception_smoke/right_angle_left_deterministic_yolo_marker_temporal
```

Result:

```text
state_source=carla_front_rgb_yolo_marker_distance
perception_adapter=ModelBackedCarlaPerceptionAdapter
visual_marker_blueprint=vehicle.mini.cooper_s
visual_marker_class=car
marker_detected=True
marker_distance_source=yolo_bbox_pinhole
```

Important: the deterministic policy was tuned for oracle/fixed-point distance,
so this smoke currently validates the model-state path rather than final task
success.

## TriLiteNet Lane Perception Smoke

A-plan lane replacement components:

```text
third_party/trilitenet
models/trilitenet/small.pth
perception/trilitenet_lane_adapter.py
experiments/run_trilitenet_lane_smoke.py
```

Offline smoke:

```text
path=runs/model_perception_smoke/trilitenet_lane_offline_smoke
state_source=front_rgb_trilitenet_lane
perception_adapter=TriLiteNetLanePerceptionAdapter
lane_center_offset_m=0.0078
heading_error_rad=-0.3323
lane_model_confidence=0.4568
drivable_area_confidence=0.8191
```

New CARLA runtime mode:

```bash
--perception-source trilitenet_lane_yolo_marker
```

This mode combines TriLiteNet lane posture with the existing YOLO marker
distance adapter. Oracle lane values are kept only in trace columns for error
analysis.

Attempted closed-loop output path:

```text
runs/model_perception_smoke/right_angle_left_deterministic_trilitenet_lane_yolo_marker
```

RPC was fixed by running the CARLA Python client in the same local-socket
enabled context as the server. Healthcheck:

```text
scripts/check_carla_rpc.sh
server_version=0.9.15
world=Carla/Maps/Town10HD_Opt
```

Closed-loop smoke after the fix:

```text
path=runs/model_perception_smoke/right_angle_left_deterministic_trilitenet_lane_yolo_marker_rpc_fixed
perception_source=trilitenet_lane_yolo_marker
state_source=carla_front_rgb_trilitenet_lane_yolo_marker_distance
attempt1_success_rate=0.0
retry_rate=1.0
final_success_rate=0.0
valid_code_rate=1.0
attempt1_feedback_counts={"turn_too_early": 1}
final_feedback_counts={"turn_too_late": 1}
mean_final_max_lane_center_offset_m=0.1427
```

This validates the closed-loop integration. The remaining failure is semantic:
the deterministic retry policy waits too long after the first feedback when the
trigger distance comes from the visual marker estimator.

Semantic calibration fixed that remaining failure:

```text
path=runs/model_perception_smoke/right_angle_left_deterministic_trilitenet_lane_yolo_marker_semantic_fixed_v3
perception_source=trilitenet_lane_yolo_marker
state_source=carla_front_rgb_trilitenet_lane_yolo_marker_distance
attempt1_success_rate=0.0
retry_rate=1.0
final_success_rate=1.0
valid_code_rate=1.0
attempt1_feedback_counts={"turn_too_early": 1}
final_feedback_counts={"none": 1}
mean_final_max_lane_center_offset_m=1.4899
```

Successful final policy sequence:

```text
hold_center x10 -> hard_left(-0.255) -> hold_left(-0.255) -> return_center -> hold_center
```

## Historical Outputs

The following directories are useful for comparison/debugging, but are not the
current canonical report paths:

```text
runs/carla_stage1_auto_feedback_latest_v2
runs/carla_stage1_auto_feedback_repeats_llm3
runs/carla_stage1_conditional_policy
runs/carla_driving_test_demos
runs/carla_feedback_retry_demo
runs/robot_skill_adapter_smoke/right_angle_left_deterministic
runs/robot_skill_adapter_smoke/right_angle_left_deterministic_smooth
```
