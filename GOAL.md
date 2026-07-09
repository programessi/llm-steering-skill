# Stage 1 Goal: Perception-Conditioned LLM Driving Skill Program

## Objective

Build a stage-1 closed-loop driving demo where a lightweight front-view perception stack updates structured road-state variables, an LLM generates restricted Python policy code over those variables, and the generated code controls a simulated vehicle through steering-skill primitives.

The stage-1 system must prove this chain:

```text
front-view sensor input
  -> reusable perception model + geometric postprocess
  -> DrivingState with confidence
  -> LLM-generated feedback-aware policy code
  -> oracle SteeringSkill interface
  -> simulated vehicle control
  -> ExecutionFeedback
  -> repeated correction attempts
```

Stage 1 does not train or run the humanoid valve skill. It must preserve the steering-skill interface so stage 2 can replace the oracle steering executor with a ManiSkill valve/wheel policy without changing LLM policy code.

## Main Constraint

Use simple sensors only:

```text
required: one front RGB camera
optional: one front depth source, either simulator depth, RGB-D, Depth Anything V2, UniDepth, or Metric3D
```

Do not build a full autonomous-driving sensor suite. Avoid multi-camera BEV, LiDAR/radar fusion, 360-degree perception, and lane-change tasks that require side/rear observability.

## Recommended Simulator

Primary target:

```text
CARLA
```

Fallback target:

```text
MetaDrive
```

The initial implementation may include an adapter abstraction so the policy runtime does not depend on the simulator. Oracle simulator state may be used only for upper-bound baselines, labels, success checks, and debugging.

## Perception Stack

Use existing models rather than training perception from scratch.

Recommended first choices:

```text
lane / drivable area:
  TriLiteNet, TwinLiteNetPlus, or another recent single-front-view road segmentation model

object detection:
  YOLO11 or another maintained detector

depth:
  simulator depth, RGB-D depth, Depth Anything V2, UniDepth, or Metric3D

tracking:
  simple Kalman filter, SORT, ByteTrack, or frame-to-frame association
```

The perception stack outputs structured variables, not driving actions.

## DrivingState v1

The first version should expose only fields that are observable from front-view sensors:

```python
@dataclass
class Estimated:
    value: object
    confidence: float
    timestamp: float

@dataclass
class DrivingState:
    speed_mps: Estimated
    steering_angle_rad: Estimated

    lane_center_offset_m: Estimated
    heading_error_rad: Estimated
    lane_curvature: Estimated
    drivable_area_confidence: Estimated

    front_vehicle_exists: Estimated
    front_vehicle_distance_m: Estimated
    front_vehicle_relative_speed_mps: Estimated

    route_command: Estimated          # keep_lane, turn_left, turn_right
    distance_to_maneuver_m: Estimated

    perception_confidence: Estimated
```

Do not put `left_lane_safe` or `right_lane_safe` into the main stage-1 task set. With only front-view sensing, side/rear safety is not reliably observable.

## LLM Policy Contract

The LLM generates restricted Python code. It must not access raw images, simulator internals, robot joints, or vehicle control channels directly.

Allowed observation calls:

```python
observe_driving_state()
observe_execution_feedback()
task_finished()
```

Allowed steering and driving primitives:

```python
steer_to(angle: float, duration: float)
hold_steering(angle: float, duration: float)
return_center(duration: float)

keep_lane_center()
recover_from_offset()
follow_front_vehicle()
prepare_turn(direction: str)
execute_turn(direction: str)
slow_down()
maintain_speed()
```

The generated policy should be a feedback-aware loop, for example:

```python
def policy():
    while not task_finished():
        state = observe_driving_state()
        feedback = observe_execution_feedback()

        if state.perception_confidence.value < 0.6:
            slow_down()
            return_center(duration=0.3)
            continue

        if feedback.event == "turn_too_early":
            return_center(duration=0.3)
            continue

        if abs(state.lane_center_offset_m.value) > 0.25:
            recover_from_offset()
            continue

        if state.front_vehicle_exists.value and state.front_vehicle_distance_m.value < 10.0:
            follow_front_vehicle()
            continue

        if state.route_command.value in ("turn_left", "turn_right"):
            prepare_turn(state.route_command.value)
            continue

        keep_lane_center()
```

## SteeringSkill v1

Stage 1 uses an oracle implementation behind the same interface that stage 2 will use:

```python
class SteeringSkill:
    def steer_to(self, target_angle: float, duration: float) -> SkillResult:
        ...

    def hold(self, angle: float, duration: float) -> SkillResult:
        ...

    def return_center(self, duration: float) -> SkillResult:
        ...
```

The stage-1 oracle skill maps target wheel angle to simulator vehicle steering. Stage 2 will replace this with a ManiSkill valve/wheel executor that returns measured wheel angle.

## ExecutionFeedback v1

Feedback must be explicit and queryable by LLM-generated code:

```python
@dataclass
class ExecutionFeedback:
    last_skill_name: str | None
    last_skill_success: bool
    target_angle: float | None
    actual_steering_angle: float | None
    lane_error_after_skill_m: float | None
    heading_error_after_skill_rad: float | None
    event: str | None
```

Initial event labels:

```text
under_steer
over_steer
turn_too_early
turn_too_late
lane_departure_risk
front_vehicle_too_close
perception_low_confidence
none
```

## Stage-1 Tasks

Implement and evaluate these tasks first:

1. Lane Keeping
   Keep the vehicle centered in a lane for a fixed horizon.

2. Offset Recovery
   Start with left/right lateral offset and recover to lane center through repeated steering-skill calls.

3. Curve Following
   Follow a curved road using lane curvature, lane offset, and heading error.

4. Front Vehicle Following
   Detect a slow front vehicle and maintain safe distance by slowing down or stabilizing steering.

5. Route-Conditioned Turning
   Given a route command and distance-to-maneuver estimate, turn left or right at the correct time and recover if the turn starts too early or too late.

## Baselines

Run at least these comparisons:

```text
oracle state + rule policy
oracle state + LLM code policy
perceived state + rule policy
perceived state + LLM code policy
perceived state + LLM code policy + feedback recovery
```

The central comparison is not whether the LLM can output one good steering angle. It is whether feedback-aware generated code can recover from perception noise and execution errors over multiple attempts.

## Metrics

Report:

```text
task success rate
collision rate
mean lane-center offset
max lane-center offset
front-vehicle minimum distance
turn-too-early / turn-too-late count
low-confidence conservative-action success rate
valid generated-code rate
primitive call count
feedback-recovery success rate
```

Perception-specific metrics:

```text
lane-center offset error against oracle
heading error against oracle
front-vehicle distance error against oracle
front-vehicle detection precision / recall
confidence calibration
```

## Deliverables

Minimum files to implement:

```text
stage1_closed_loop_driving/
  GOAL.md
  docs/
    architecture.md
    perception_adapter_notes.md
  perception/
    state_schema.py
    perception_adapter.py
    oracle_state_adapter.py
  llm_policy/
    prompts/
      stage1_policy_prompt.md
    code_generator.py
    policy_runtime.py
    safety_checker.py
  skills/
    steering_skill.py
    oracle_steering_skill.py
    feedback.py
  envs/
    simulator_adapter.py
    carla_adapter.py
    metadrive_adapter.py
  experiments/
    run_stage1_demo.py
    run_stage1_baselines.py
    run_perception_eval.py
  runs/
```

Adapters may initially be partial if a simulator is not installed, but the interfaces and one runnable closed-loop demo must exist.

## Completion Criteria

Stage 1 is complete when:

```text
1. A front-view perception adapter produces DrivingState v1 fields.
2. LLM-generated restricted Python policy code runs in the policy runtime.
3. The policy controls a simulated vehicle only through SteeringSkill primitives.
4. ExecutionFeedback is generated and consumed inside the policy loop.
5. Lane keeping, offset recovery, curve following, front-vehicle following, and route-conditioned turning are runnable tasks.
6. Results are saved under runs/ with metrics and at least one video or trajectory visualization.
7. Baselines show the gap between oracle state, perceived state, and feedback-aware LLM policy.
8. The oracle SteeringSkill can be replaced by a future ManiSkill valve/wheel skill without changing the generated policy code.
```

## Suggested Goal-Mode Objective

Use this as the Codex goal objective:

```text
Implement Stage 1 of the humanoid-driving-valve project: build a simple-front-sensor closed-loop driving demo where reusable perception models or adapters produce DrivingState, an LLM generates restricted feedback-aware Python policy code, and the policy controls a CARLA/MetaDrive vehicle only through oracle SteeringSkill primitives, with runnable tasks, baselines, metrics, and documentation under /home/xingshu/workspaces/fys/stage1_closed_loop_driving.
```
