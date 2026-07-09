# Stage 1 Architecture

Stage 1 validates the software contract before the humanoid valve skill is
inserted.

```text
front RGB/depth sensor
  -> perception adapter
  -> DrivingState
  -> restricted LLM policy runtime
  -> SteeringSkill primitives
  -> simulator adapter
  -> ExecutionFeedback
```

The current workspace does not have CARLA or MetaDrive installed, so the
runnable smoke path uses `KinematicDrivingSimulator`. It renders a front-view
RGB/depth observation and updates a simple lane-relative vehicle model. This is
only a fallback adapter; the policy runtime depends on `SimulatorAdapter`, so a
CARLA or MetaDrive adapter can replace it later.

The current mainline CARLA agent uses real CARLA front RGB for video output,
but its `DrivingState` fields are still CARLA/map oracle values plus a
fixed-point distance-to-maneuver adapter. The perception module is therefore a
replaceable state adapter, not a learned road-understanding model yet.

The important boundary is that LLM-generated code cannot control the vehicle or
the robot directly. The model sees compact API descriptions in the prompt, not
the full implementation of those functions. In the CARLA conditional policy
runner, generated code can only call:

```python
observe_driving_state()
observe_execution_feedback()
task_finished()
mark_task_complete()
execute_primitive(name, target_angle_rad, duration_s, speed_mps, stage="...", trigger="...")
```

`execute_primitive(...)` is the only action outlet. Today it calls
`OracleSteeringSkill` in CARLA; Stage 2 should replace that implementation with
a ManiSkill/robot valve-or-wheel skill that accepts the same target-angle and
duration command and returns measured wheel-angle feedback. The generated policy
code should not change.
