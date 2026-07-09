# Stage 1 Status

Implemented:

```text
DrivingState and Estimated schema
ExecutionFeedback and SkillResult schema
restricted policy runtime
rule / no-feedback / feedback-aware policy modes
oracle SteeringSkill interface
front-view CV perception adapter
oracle state adapter
local kinematic simulator fallback
MetaDrive adapter smoke path
CARLA adapter and rendered demo runner
single-task demo runner
baseline runner
perception eval runner
front-view video export
discrete steering primitive trace and metrics
CARLA rendered demo runner
CARLA driving-test primitive demo runner
```

Latest verification:

```bash
python stage1_closed_loop_driving/experiments/run_stage1_baselines.py \
  --out stage1_closed_loop_driving/runs/stage1_baselines_discrete_final \
  --video-task curve_following

python stage1_closed_loop_driving/experiments/run_perception_eval.py \
  --out stage1_closed_loop_driving/runs/perception_eval.json

env MPLCONFIGDIR=/tmp/mpl METADRIVE_LOG_LEVEL=50 \
  /home/xingshu/workspaces/fys/stage1_closed_loop_driving/.venv310/bin/python \
  stage1_closed_loop_driving/experiments/run_metadrive_smoke.py \
  --out stage1_closed_loop_driving/runs/metadrive_smoke \
  --horizon 120

env SDL_VIDEODRIVER=dummy MPLCONFIGDIR=/tmp/mpl METADRIVE_LOG_LEVEL=50 \
  /home/xingshu/workspaces/fys/stage1_closed_loop_driving/.venv310/bin/python \
  stage1_closed_loop_driving/experiments/run_metadrive_rendered_demo.py \
  --out stage1_closed_loop_driving/runs/metadrive_rendered_demo \
  --horizon 160 \
  --width 960 \
  --height 540 \
  --fps 20

/home/xingshu/workspaces/fys/stage1_closed_loop_driving/.venv310/bin/python \
  stage1_closed_loop_driving/experiments/run_carla_rendered_demo.py \
  --out stage1_closed_loop_driving/runs/carla_rendered_demo \
  --host 127.0.0.1 \
  --port 2000 \
  --town Town04 \
  --spawn-index 0 \
  --horizon 240 \
  --width 1280 \
  --height 720 \
  --fps 20

stage1_closed_loop_driving/scripts/run_carla_driving_test_demos.sh
```

Observed behavior:

```text
Final fallback baseline: 34/42 successful.

llm_feedback + oracle: 7/7
llm_no_feedback + oracle: 7/7
llm_feedback + perceived: 5/7
llm_no_feedback + perceived: 5/7
rule + oracle: 5/7
rule + perceived: 5/7

LLM-style no-feedback and feedback policies complete all implemented tasks when
`state_source=oracle` in the local fallback simulator. This is the clean
software-contract test for:

DrivingState -> restricted policy code -> discrete SteeringSkill primitives.

The fallback `perceived` path uses a toy OpenCV lane detector over synthetic
front RGB, plus a route/map curvature provider because the local renderer does
not draw true road curvature into the image. It completes lane keeping, offset
recovery, curve following, and front-vehicle following. It still fails the
route-turn tasks after the maneuver because the toy lane detector loses
reliable lateral state in the coarse turning geometry. This is a
perception-adapter limitation, not a steering-skill interface limitation.

The rule policy completes lane keeping, offset recovery, curve following, and
front-vehicle following, but fails the route-turn tasks. This provides an
initial baseline gap for the route-conditioned skill-program logic.

Each `trace.csv` now includes:

```text
skill_name
steering_primitive
target_angle_rad
```

Each `summary.json` now includes:

```text
steering_primitive_counts
steering_primitive_sequence
```

MetaDrive lane-keeping smoke succeeds through the same `OracleSteeringSkill`
interface:

```text
steps=98
success=true
max_lane_center_offset_m=0.486
primitive_call_count=26
```

MetaDrive rendered demo output:

```text
stage1_closed_loop_driving/runs/metadrive_rendered_demo/metadrive_rendered_demo.mp4
stage1_closed_loop_driving/runs/metadrive_rendered_demo/preview.png
stage1_closed_loop_driving/runs/metadrive_rendered_demo/summary.json
```

CARLA driving-test primitive demo output:

```text
stage1_closed_loop_driving/runs/carla_driving_test_demos/summary.json
demo_count=3
success_count=3
steering commands ramp over each stage duration
demo speeds are reduced to roughly 2.2-4.0 m/s
each demo writes front_rgb.mp4, preview.png, trace.csv, summary.json,
policy.py, and steering_curve.png

right_angle_left:
  sequence = hold_center -> hard_left -> hold_left -> return_center -> hold_center

lane_change_left:
  sequence = hold_center -> medium_left -> medium_right -> return_center -> hold_center

pull_over_right:
  sequence = hold_center -> medium_right -> medium_left -> return_center -> hold_center_stop
```

Current experiment boundary:

```text
camera_frame=carla_front_rgb
state_source=carla_map_oracle
perception_adapter=CarlaMapOracleStateAdapter
control_source=SteeringSkill
steering_skill_impl=OracleSteeringSkill
```

This means the video is CARLA camera render, but the policy's structured
driving variables are still simulator/map truth rather than a learned visual
perception model. That boundary is now recorded both in `summary.json` and in
every `trace.csv` row for the CARLA adapter.

CARLA feedback retry demo:

```text
stage1_closed_loop_driving/scripts/run_carla_feedback_retry_demo.sh
stage1_closed_loop_driving/runs/carla_feedback_retry_demo/summary.json

attempt_01_turn_too_early:
  LLM-style policy turns in too early.
  feedback event = turn_too_early
  success = false

attempt_02_feedback_repaired:
  LLM-style policy delays the trigger, reduces peak steering, and extends
  return-center duration.
  feedback event = none
  success = true if the raw CARLA run stays inside the lane-departure limit
```

CARLA Stage-1 experiment table:

```text
stage1_closed_loop_driving/scripts/run_carla_stage1_experiment_table.sh
stage1_closed_loop_driving/runs/carla_stage1_experiment_table/results.csv
stage1_closed_loop_driving/runs/carla_stage1_experiment_table/summary.json
```

The table runner currently focuses on the right-angle-left task and compares:

```text
fixed_no_feedback:
  generic open-loop fixed-point timing, no feedback repair

llm_no_feedback:
  task-conditioned generated policy, but only one attempt

llm_feedback_retry:
  task-conditioned generated policy, then a second generated policy after
  receiving turn_too_early feedback
```

The policy generator is deterministic for repeatability. It explicitly models
the future LLM interface:

```text
task spec + DrivingState schema + ExecutionFeedback -> structured policy code
```

The generated `policy.py` files still call only `execute_primitive(...)`, so
continuous steering remains inside the replaceable `SteeringSkill`.

Latest Stage-1 experiment table result:

```text
results.csv:

fixed_no_feedback:
  success=0
  execution_feedback_event=turn_too_early
  mean_lane_center_offset_m=1.7454
  max_lane_center_offset_m=3.6997

llm_no_feedback:
  success=0
  execution_feedback_event=turn_too_early
  mean_lane_center_offset_m=1.1047
  max_lane_center_offset_m=3.4940

llm_feedback_retry:
  success=1
  execution_feedback_event=none
  retry_attempts=1
  feedback_corrections=1
  mean_lane_center_offset_m=0.5972
  max_lane_center_offset_m=2.5022
```

Next executable conditional policy runner:

```text
stage1_closed_loop_driving/scripts/run_carla_stage1_conditional_policy.sh
stage1_closed_loop_driving/runs/carla_stage1_conditional_policy/results.csv
```

This is the first runner where the generated `policy.py` is actually executed
as code instead of being treated as a static stage list. The policy can:

```text
observe_driving_state()
observe_execution_feedback()
while not task_finished()
if feedback.event == ...
if state.distance_to_maneuver_m ...
execute_primitive(...)
```

The current fixed-point `distance_to_maneuver_m` is still a CARLA oracle-derived
task variable, not learned perception.

Latest executable conditional policy result:

```text
results.csv:

static_sequence_no_feedback:
  success=0
  execution_feedback_event=turn_too_early
  primitive_call_count=5
  mean_lane_center_offset_m=1.1206
  max_lane_center_offset_m=3.5064

static_sequence_feedback_retry:
  success=1
  execution_feedback_event=none
  retry_attempts=1
  feedback_corrections=1
  primitive_call_count=5
  mean_lane_center_offset_m=0.5942
  max_lane_center_offset_m=2.4704

conditional_policy_feedback_retry:
  success=1
  execution_feedback_event=none
  retry_attempts=1
  feedback_corrections=1
  primitive_call_count=15
  mean_lane_center_offset_m=0.4137
  max_lane_center_offset_m=1.9201
```

Robot steering skill bench:

```text
stage1_closed_loop_driving/scripts/run_robot_steering_skill_bench.sh
stage1_closed_loop_driving/docs/robot_steering_skill_bench.md
```

Implemented:

```text
robot_steering/schema.py
robot_steering/metrics.py
robot_steering/robot_steering_skill.py
robot_steering/stub_skill.py
experiments/run_robot_steering_skill_bench.py
```

This defines the Stage-2 replacement target for `OracleSteeringSkill`:

```text
SteeringSkill command:
  current_angle_rad
  target_angle_rad
  duration_s
  hold_s

Bench metrics:
  success_rate
  final_angle_error_rad
  overshoot_rad
  settling_time_s
hold_stability_rad
```

## Simulated Robot Steering Adapter

Implemented:

```text
skills/simulated_robot_steering_skill.py
```

This adapter implements the same `SteeringSkill` methods used by generated
policy code, but internally calls a robot-side steering skill:

```text
execute_primitive(...)
  -> SteeringSkill.steer_to/hold/return_center(...)
  -> SimulatedRobotSteeringSkill
  -> StubRobotSteeringSkill.execute(SteeringSkillCommand)
  -> measured robot wheel-angle trajectory
  -> CarlaAdapter.step(actual_wheel_angle, speed)
```

The LLM policy and runtime API do not change. The replacement is selected with:

```bash
--steering-skill simulated_robot
```

Latest CARLA smoke:

```text
runs/robot_skill_adapter_smoke/right_angle_left_llm_smooth
policy_generator=llm
steering_skill_impl=SimulatedRobotSteeringSkill
robot_skill_impl=StubRobotSteeringSkill
attempt1_feedback=turn_too_early
final_feedback=none
final_success_rate=1.0
final_max_lane_center_offset_m=1.3690
```

The generated policy still writes structured code with only
`execute_primitive(...)` calls. The trace confirms the actual control path with
`robot_observed_wheel_angle_rad`, `robot_action_delta_angle_rad`, and
`robot_steering_episode_id` fields.

## Valve Backend Bridge

Implemented a first valve-as-steering backend:

```text
robot_steering/valve_skill.py
robot_steering/backends.py
```

Runtime selection:

```bash
--steering-skill simulated_robot
--robot-skill-backend kinematic_valve
```

This replaces the robot-side stub with a valve joint proxy:

```text
execute_primitive(...)
  -> SimulatedRobotSteeringSkill
  -> KinematicValveSteeringSkill
  -> valve_joint_qpos_rad
  -> CARLA steering_angle_rad
```

CARLA smoke:

```text
runs/valve_backend_smoke/right_angle_left_deterministic_kinematic_valve
robot_skill_backend=kinematic_valve
robot_skill_impl=KinematicValveSteeringSkill
valve_joint_qpos_rad == steering_angle_rad
```

ManiSkill3 source is cloned under:

```text
third_party/maniskill3
```

Relevant ManiSkill tasks:

```text
RotateValveLevel0-v1
TurnFaucet-v1
```

The next implementation target is wiring `ManiSkillValveSteeringSkill` to
`RotateValveLevel0-v1` and reading `env.valve.qpos[:, 0]` as the actual wheel
angle.

Installed and verified:

```text
mani_skill==3.0.1
sapien==3.0.3
torch==2.13.0+cu130
```

`ManiSkillValveSteeringSkill` now creates `RotateValveLevel0-v1`, writes a
valve qpos trajectory, reads back `env.unwrapped.valve.qpos[:, 0]`, and feeds
that value to CARLA through the existing `SimulatedRobotSteeringSkill`.

Smoke result:

```text
runs/valve_backend_smoke/right_angle_left_deterministic_maniskill_valve
robot_skill_backend=maniskill_valve
robot_skill_impl=ManiSkillValveSteeringSkill
trace_rows_with_valve=183
max_abs_diff_valve_vs_steer=0.0
```

## Model-Backed Perception

Implemented the first CARLA model-backed perception adapter:

```text
perception/model_backed_carla_adapter.py
models/yolo11n.pt
```

Runtime selection:

```bash
--perception-source model_marker
```

This replaces the LLM's `distance_to_maneuver_m` source:

```text
before:
  fixed-point script distance / oracle-style runtime distance

after:
  CARLA front RGB
    -> YOLO11 detects visual marker
    -> bounding-box pinhole distance
    -> DrivingState.distance_to_maneuver_m
```

The first marker-backed smoke is:

```text
runs/model_perception_smoke/right_angle_left_deterministic_yolo_marker_temporal
state_source=carla_front_rgb_yolo_marker_distance
perception_adapter=ModelBackedCarlaPerceptionAdapter
```

Trace confirms the LLM-visible state is model-derived:

```text
llm_perception_source=model_marker
perceived_distance_to_maneuver_m
oracle_distance_to_maneuver_m
perceived_distance_error_m
marker_detected=True
marker_distance_source=yolo_bbox_pinhole
```

The deterministic retry policy is not yet calibrated for this visual distance
curve. Current result can be `turn_too_late`, but oracle distance is no longer
the state source for the field used by the policy trigger.

Latest stub bench result:

```text
runs/robot_steering_skill_bench/summary.json
runs/robot_steering_skill_bench/results.csv

episode_count=5
success_count=5
success_rate=1.0
mean_final_angle_error_rad=0.001112
max_final_angle_error_rad=0.002376
mean_overshoot_rad=0.0
mean_settling_time_s=0.49
hold_stability_rad=0.000753
```

OpenAI-compatible LLM policy generator:

```text
llm_policy/openai_compatible_client.py
llm_policy/prompt_builder.py
llm_policy/llm_code_generator.py
experiments/generate_llm_policy_smoke.py
scripts/generate_llm_policy_smoke.sh
```

Configuration:

```text
STAGE1_LLM_BASE_URL or OPENAI_BASE_URL
STAGE1_LLM_API_KEY or OPENAI_API_KEY
STAGE1_LLM_MODEL or OPENAI_MODEL
```

The prompt is modeled after the CaP-X R1Pro configs: it gives a compact API
reference and asks for one continuous executable Python script defining
`policy()`. It does not expose full internal function implementations to the
model. The OpenAI-compatible generator now validates generated code with an AST
checker before execution and automatically reprompts once with the validation
error if the code tries to use imports, helper functions, unlisted calls, or an
action path other than `execute_primitive(...)`.

The CARLA conditional runner now accepts:

```bash
scripts/run_carla_stage1_conditional_policy.sh --policy-generator llm
```

Latest real LLM policy generation result:

```text
scripts/generate_llm_policy_smoke.sh
  model=gpt-5.5
  base_url=https://ai.zxcoding.top/v1
  compiled=true

scripts/run_carla_stage1_conditional_policy.sh --policy-generator llm

static_sequence_no_feedback:
  success=0
  max_lane_center_offset_m=3.4800

static_sequence_feedback_retry:
  success=1
  max_lane_center_offset_m=2.5105

conditional_policy_feedback_retry:
  policy_generator=openai_compatible_llm_generator
  success=1
  primitive_call_count=18
  mean_lane_center_offset_m=0.1377
  max_lane_center_offset_m=0.7139
```

Latest automatic-feedback CARLA rerun:

```text
scripts/run_carla_stage1_conditional_policy.sh --policy-generator llm \
  --out runs/carla_stage1_auto_feedback_latest_v2

conditional_policy_feedback_retry attempt_01:
  feedback_source=auto_trace_metrics
  execution_feedback_event=turn_too_early
  success=0

conditional_policy_feedback_retry attempt_02:
  llm_input_feedback=turn_too_early: delay trigger, reduce peak steering, return center later
  feedback_source=auto_trace_metrics
  execution_feedback_event=none
  success=1
  primitive_call_count=19
  mean_lane_center_offset_m=0.2044
  max_lane_center_offset_m=1.0519
```

Automatic-feedback repeat experiment:

```text
scripts/run_carla_stage1_auto_feedback_repeats.sh --trials 3 --policy-generator llm \
  --out runs/carla_stage1_auto_feedback_repeats_llm3

trials=3
attempt1_success_rate=0.0
retry_rate=1.0
final_success_rate=1.0
valid_code_rate=1.0
attempt1_feedback_counts={"turn_too_early": 3}
final_feedback_counts={"none": 3}
mean_final_max_lane_center_offset_m=1.0235
```

Second fixed-point task:

```text
scripts/run_carla_stage1_auto_feedback_repeats.sh --task lane_change_left \
  --trials 3 --policy-generator llm \
  --out runs/auto_feedback_tasks/lane_change_left/repeats_llm3

trials=3
attempt1_success_rate=0.0
retry_rate=1.0
final_success_rate=1.0
valid_code_rate=1.0
attempt1_feedback_counts={"turn_too_early": 3}
final_feedback_counts={"none": 3}
mean_final_max_lane_center_offset_m=0.7024
```

The successful lane-change policies use a different steering sequence from the
right-angle turn, typically:

```text
hold_center -> medium_left -> hold_left -> medium_right -> return_center -> hold_center
```

The first LLM prompt allowed `trigger_distance_m >= 8.0`, so the generated code
could turn immediately. The prompt was tightened to state that the benchmark
starts near `distance_to_maneuver_m=8.0`, requires initial
`trigger_distance_m` in `[4.5, 6.0]`, and requires `turn_too_early` repair to
reduce it into `[3.0, 4.2]`. The generated policy now waits with repeated
`hold_center` calls before turning.

Known limitations:

```text
The current perception adapter is classic CV over a synthetic front view, not a
pretrained TriLiteNet/TwinLiteNet/YOLO11 stack yet.

The `perceived` route-turn failures are the clearest evidence that the next
meaningful improvement is replacing the toy CV adapter with a stronger reusable
driving perception stack or using a simulator with a proper camera/lane
annotation feed.

`CarlaAdapter`, `run_carla_rendered_demo.py`, and
`run_carla_driving_test_demos.py` are implemented. CARLA 0.9.15 is installed
under `third_party/carla`, and the Python client in `.venv310` can connect to
the local server when run with local socket permission.

MetaDrive is installed in the project-local Python 3.10 venv:

```text
stage1_closed_loop_driving/.venv310
```

The current MetaDrive adapter uses oracle lane state. It does not yet return a
real front RGB camera frame into `FrontViewCVPerceptionAdapter`.

The rendered demo uses MetaDrive's top-down renderer as the video source and
routes that frame through `MetaDriveRenderedStateAdapter`. Direct `RGBCamera`
offscreen capture currently fails in this headless environment with a
Panda3D/simplePBR tonemapping initialization error:

```text
AttributeError: 'NoneType' object has no attribute 'set_shader'
```

To use real front RGB frames, run on a machine/session with working Panda3D
offscreen rendering, EGL, or an onscreen display, then replace
`MetaDriveRenderedStateAdapter` with a model-backed RGB perception adapter.

The route-turn geometry in the local fallback simulator is intentionally coarse.
It is useful for exercising interfaces, not for final driving realism.
```

## 2026-07-09 A-Plan Perception Update

Implemented the first lane-posture model replacement:

```text
perception/trilitenet_lane_adapter.py
experiments/run_trilitenet_lane_smoke.py
```

Added runtime mode:

```bash
--perception-source trilitenet_lane_yolo_marker
```

Field ownership in this mode:

```text
lane_center_offset_m / heading_error_rad / lane_curvature
  from TriLiteNet lane + drivable-area segmentation over CARLA front RGB

distance_to_maneuver_m
  from YOLO11 visual marker distance

oracle lane and oracle distance
  trace-only comparison fields, not LLM-visible state
```

Offline smoke passed:

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv310/bin/python \
  experiments/run_trilitenet_lane_smoke.py \
  --out runs/model_perception_smoke/trilitenet_lane_offline_smoke
```

Observed output:

```text
lane_center_offset_m=0.0078
heading_error_rad=-0.3323
lane_model_confidence=0.4568
drivable_area_confidence=0.8191
```

CARLA closed-loop smoke initially appeared blocked by the simulator:

```text
RuntimeError: time-out of 60000ms while waiting for the simulator
CARLA UE4 exited with Signal 11 when stopped
```

Root cause and fix:

```text
The CARLA server was launched with elevated/local graphics permissions, but the
Python client was still run from a restricted sandbox without local socket
permission. Running the client in the same local-socket-enabled context fixes
RPC.
```

Healthcheck:

```bash
scripts/check_carla_rpc.sh
```

Observed:

```text
server_version=0.9.15
world=Carla/Maps/Town10HD_Opt
```

Closed-loop TriLiteNet lane + YOLO marker smoke after the fix:

```text
path=runs/model_perception_smoke/right_angle_left_deterministic_trilitenet_lane_yolo_marker_rpc_fixed
perception_source=trilitenet_lane_yolo_marker
state_source=carla_front_rgb_trilitenet_lane_yolo_marker_distance
valid_code_rate=1.0
attempt1_feedback=turn_too_early
final_feedback=turn_too_late
mean_final_max_lane_center_offset_m=0.1427
```

Trace confirms LLM-visible state came from the model path:

```text
llm_perception_source=trilitenet_lane_yolo_marker
lane_model_source=trilitenet_segmentation
lane_model_config=small
marker_distance_source=yolo_bbox_pinhole
```

The semantic failure was then corrected by calibrating feedback retry parameters
for model-estimated marker distance:

```text
path=runs/model_perception_smoke/right_angle_left_deterministic_trilitenet_lane_yolo_marker_semantic_fixed_v3
final_success_rate=1.0
final_feedback_counts={"none": 1}
mean_final_max_lane_center_offset_m=1.4899
```

The key change was not a new controller. It was a different retry policy for
`turn_too_early` under visual marker distance:

```text
trigger_distance_m: 5.2 -> 3.75
turn_angle_rad: -0.34 -> -0.255
turn_duration_s: 1.7 -> 1.4
hold_duration_s: 1.5 -> 0.7
return_duration_s: 0.9 -> 1.5
```

Successful final trace:

```text
first maneuver runtime_distance_to_maneuver_m=4.0526
perceived_distance_to_maneuver_m=3.5934
max_lane_center_offset_m=1.4899
llm_perception_source=trilitenet_lane_yolo_marker
lane_model_source=trilitenet_segmentation
marker_distance_source=yolo_bbox_pinhole
```
