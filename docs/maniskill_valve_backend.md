# ManiSkill Valve Backend

This note tracks the valve-as-steering bridge.

## Current Bridge

Implemented:

```text
robot_steering/valve_skill.py
robot_steering/backends.py
skills/simulated_robot_steering_skill.py
```

Runtime selection:

```bash
scripts/run_carla_stage1_auto_feedback_repeats.sh \
  --task right_angle_left \
  --trials 1 \
  --policy-generator deterministic \
  --steering-skill simulated_robot \
  --robot-skill-backend kinematic_valve \
  --out runs/valve_backend_smoke/right_angle_left_deterministic_kinematic_valve
```

The current `kinematic_valve` backend is not a robot policy. It is a bridge
test that converts a `SteeringSkillCommand` into a simulated valve joint
trajectory, then feeds the measured valve angle back into CARLA:

```text
execute_primitive(...)
  -> SimulatedRobotSteeringSkill
  -> KinematicValveSteeringSkill
  -> valve_joint_qpos_rad
  -> CarlaAdapter.step(valve_joint_qpos_rad, speed)
```

Verified trace fields:

```text
robot_skill_backend=kinematic_valve
robot_skill_impl=KinematicValveSteeringSkill
valve_joint_qpos_rad
valve_angle_error_rad
robot_action_delta_angle_rad
```

In the deterministic smoke, `valve_joint_qpos_rad` and CARLA
`steering_angle_rad` match exactly.

## Installed Environment

Installed into:

```text
.venv310
```

Verified packages:

```text
mani_skill==3.0.1
sapien==3.0.3
torch==2.13.0+cu130
```

`RotateValveLevel0-v1` can be created on the host with device access, and
`env.unwrapped.valve.qpos` can be read.

## ManiSkill3 Source

ManiSkill3 source is stored locally at:

```text
third_party/maniskill3
```

Revision:

```text
42b6824
```

Relevant tasks found in the source tree:

```text
mani_skill/envs/tasks/dexterity/rotate_valve.py
  RotateValveLevel0-v1
  RotateValveLevel1-v1
  RotateValveLevel2-v1
  RotateValveLevel3-v1
  RotateValveLevel4-v1

mani_skill/envs/tasks/tabletop/turn_faucet.py
  TurnFaucet-v1
```

The preferred next target is `RotateValveLevel0-v1`, because its environment
already exposes the valve articulation as `self.valve` and reads the joint
state through `self.valve.qpos`.

## Real ManiSkill Backend

Implemented:

```text
ManiSkillValveSteeringSkill
```

Runtime selection:

```bash
scripts/run_carla_stage1_auto_feedback_repeats.sh \
  --task right_angle_left \
  --trials 1 \
  --policy-generator deterministic \
  --steering-skill simulated_robot \
  --robot-skill-backend maniskill_valve \
  --out runs/valve_backend_smoke/right_angle_left_deterministic_maniskill_valve
```

Verified:

```text
path=runs/valve_backend_smoke/right_angle_left_deterministic_maniskill_valve
robot_skill_backend=maniskill_valve
robot_skill_impl=ManiSkillValveSteeringSkill
trace_rows_with_valve=183
max_abs_diff_valve_vs_steer=0.0
```

This backend still directly sets the ManiSkill valve qpos. It is not yet a
learned DClaw/Panda policy. It is the installed-physics version of the bridge:

```text
SteeringSkillCommand.target_angle_rad
  -> ManiSkill RotateValveLevel0-v1 valve qpos trajectory
  -> env.unwrapped.valve.qpos[:, 0]
  -> SteeringObservation.wheel_angle_rad
  -> CARLA steering input
```

## Next Wiring Point

Replace the current placeholder:

```text
direct valve qpos update inside ManiSkillValveSteeringSkill
```

with robot action execution:

```text
SteeringSkillCommand.target_angle_rad
  -> target valve qpos
  -> ManiSkill robot/controller or learned policy steps
  -> env.valve.qpos[:, 0]
  -> SteeringObservation.wheel_angle_rad
  -> CARLA steering input
```

The upper LLM policy and `execute_primitive(...)` API should remain unchanged.
