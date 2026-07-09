# Robot Steering Skill Bench

This bench defines the Stage-2 replacement target for `OracleSteeringSkill`.
The policy layer should keep calling command-level primitives, while the low
level implementation changes from simulator oracle control to a robot steering
skill.

## Skill API

```python
steer_to(target_angle_rad, duration_s)
hold(angle_rad, duration_s)
return_center(duration_s)
```

The general command record is:

```text
skill_name
current_angle_rad
target_angle_rad
duration_s
hold_s
max_speed_rad_s
```

## Demonstration Data

Each episode should store:

```text
command:
  skill_name
  current_angle_rad
  target_angle_rad
  duration_s
  hold_s

observations:
  timestamp_s
  wheel_angle_rad
  target_angle_rad
  wrist_image_path
  external_image_path
  proprio

actions:
  timestamp_s
  action_type
  values

labels:
  success
  final_angle_error_rad
```

Recommended `action_type` choices:

```text
joint_position
joint_delta
ee_delta_pose
ee_velocity
```

## Collection Protocol

Start with a bench that is not connected to a real vehicle:

```text
robot arm + steering wheel or valve mockup + angle sensor or visual marker
```

Collect short atomic demonstrations:

```text
0.00 -> -0.30
-0.30 -> 0.00
0.00 -> 0.20
0.20 -> -0.10
-0.10 -> -0.10 hold
```

Vary:

```text
initial angle
target angle
duration
hand contact pose
wheel friction
small visual occlusion
```

## Metrics

```text
success_rate
final_angle_error_rad
overshoot_rad
settling_time_s
hold_stability_rad
```

The first pass should optimize skill reliability, not driving performance.
The driving policy experiment already treats this skill as replaceable.

## Run Stub Bench

```bash
stage1_closed_loop_driving/scripts/run_robot_steering_skill_bench.sh
```

Outputs:

```text
runs/robot_steering_skill_bench/summary.json
runs/robot_steering_skill_bench/results.csv
runs/robot_steering_skill_bench/episodes/*/sample.json
runs/robot_steering_skill_bench/episodes/*/trajectory.csv
```

