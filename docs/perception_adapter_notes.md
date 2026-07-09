# Perception Adapter Notes

The stage-1 perception contract is:

```text
front sensor observation -> DrivingState
```

The runnable local adapter is `FrontViewCVPerceptionAdapter`. It estimates lane
offset and heading from rendered front-view lane lines, and estimates front-car
distance from the optional depth image.

For the real stage-1 stack, replace this adapter with a model-backed adapter:

```text
TriLiteNet/TwinLiteNetPlus -> lane and drivable area masks
YOLO11 -> front vehicle detection
Depth Anything V2 / UniDepth / RGB-D -> front vehicle distance
tracker -> relative speed
geometric postprocess -> DrivingState fields
```

The LLM policy runtime consumes only `DrivingState`, so perception backends are
swappable.

## CARLA Model Marker Adapter

Implemented first non-oracle CARLA perception path:

```text
perception/model_backed_carla_adapter.py
```

Runtime selection:

```bash
--perception-source model_marker
```

The first replaced field is:

```text
distance_to_maneuver_m
```

It is estimated from:

```text
CARLA front RGB
  -> YOLO11 marker detection
  -> bounding-box pinhole distance
  -> optional temporal/odometry stabilization
```

The marker is spawned by CARLA as a visible object:

```text
visual_marker_blueprint=vehicle.mini.cooper_s
visual_marker_class=car
```

LLM-visible state source:

```text
state_source=carla_front_rgb_yolo_marker_distance
perception_adapter=ModelBackedCarlaPerceptionAdapter
```

Trace fields:

```text
llm_perception_source
perceived_distance_to_maneuver_m
oracle_distance_to_maneuver_m
perceived_distance_error_m
marker_detected
marker_confidence
marker_distance_source
```

Current smoke:

```text
runs/model_perception_smoke/right_angle_left_deterministic_yolo_marker_temporal
policy_generator=deterministic
steering_skill_impl=OracleSteeringSkill
perception_source=model_marker
marker_distance_source=yolo_bbox_pinhole
```

This smoke proves that generated policy code reads model-derived marker
distance rather than fixed-point oracle distance. The deterministic retry policy
is not yet tuned for the model-distance curve and can end in `turn_too_late`.
That is a policy/perception calibration issue, not an oracle-state path.

## TriLiteNet Lane Adapter

Added A-plan lane posture replacement:

```text
perception/trilitenet_lane_adapter.py
third_party/trilitenet
models/trilitenet/small.pth
```

Runtime selection:

```bash
--perception-source trilitenet_lane_yolo_marker
```

This combined source means:

```text
lane_center_offset_m, heading_error_rad, lane_curvature
  <- CARLA front RGB -> TriLiteNet lane/drivable-area segmentation -> geometric postprocess

distance_to_maneuver_m
  <- CARLA front RGB -> YOLO marker detection -> bbox pinhole distance
```

LLM-visible state source:

```text
state_source=carla_front_rgb_trilitenet_lane_yolo_marker_distance
perception_adapter=ModelBackedCarlaPerceptionAdapter(TriLiteNetLanePerceptionAdapter+YOLOMarker)
```

New trace fields:

```text
perceived_lane_center_offset_m
perceived_heading_error_rad
oracle_lane_center_offset_m
oracle_heading_error_rad
perceived_lane_center_offset_error_m
perceived_heading_error_rad_error
lane_model_source
lane_model_config
lane_model_lane_pixel_count
lane_model_drivable_pixel_count
lane_model_confidence
lane_model_drivable_confidence
lane_model_px_per_m
```

Offline smoke:

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv310/bin/python \
  experiments/run_trilitenet_lane_smoke.py \
  --out runs/model_perception_smoke/trilitenet_lane_offline_smoke
```

Current offline result on `runs/model_perception_smoke/marker_frame.png`:

```text
lane_center_offset_m=0.0078
heading_error_rad=-0.3323
lane_model_confidence=0.4568
drivable_area_confidence=0.8191
```

CARLA closed-loop smoke command:

```bash
MPLCONFIGDIR=/tmp/matplotlib scripts/run_carla_stage1_auto_feedback_repeats.sh \
  --task right_angle_left \
  --trials 1 \
  --policy-generator deterministic \
  --steering-skill oracle \
  --perception-source trilitenet_lane_yolo_marker \
  --visual-marker-distance-m 8.0 \
  --visual-marker-lateral-offset-m 2.2 \
  --visual-marker-real-height-m 2.0 \
  --out runs/model_perception_smoke/right_angle_left_deterministic_trilitenet_lane_yolo_marker
```

RPC fix on 2026-07-09:

```text
scripts/check_carla_rpc.sh
server_version=0.9.15
world=Carla/Maps/Town10HD_Opt
```

The earlier timeout was caused by running the CARLA Python client in a sandboxed
context without local socket permission. The CARLA server was started with
elevated permissions, but the client also needs local socket permission.

Closed-loop smoke after the RPC fix:

```text
runs/model_perception_smoke/right_angle_left_deterministic_trilitenet_lane_yolo_marker_rpc_fixed
perception_source=trilitenet_lane_yolo_marker
state_source=carla_front_rgb_trilitenet_lane_yolo_marker_distance
valid_code_rate=1.0
attempt1_feedback_counts={"turn_too_early": 1}
final_feedback_counts={"turn_too_late": 1}
mean_final_max_lane_center_offset_m=0.1427
```

The smoke validates the CARLA RPC, video/trace generation, generated policy
runtime, YOLO marker distance, and TriLiteNet lane-state path. It does not yet
pass the semantic fixed-point turn criterion; the remaining issue is
policy/perception calibration.

Semantic fix after calibration:

```text
runs/model_perception_smoke/right_angle_left_deterministic_trilitenet_lane_yolo_marker_semantic_fixed_v3
final_success_rate=1.0
final_feedback_counts={"none": 1}
mean_final_max_lane_center_offset_m=1.4899
```

The calibration changes the `turn_too_early` retry from an oracle-distance
style correction to a marker-distance-aware correction:

```text
trigger_distance_m: 5.2 -> 3.75
turn_angle_rad: -0.34 -> -0.255
turn_duration_s: 1.7 -> 1.4
hold_duration_s: 1.5 -> 0.7
return_duration_s: 0.9 -> 1.5
```

The successful final trace starts the first maneuver at:

```text
perceived_distance_to_maneuver_m=3.5934
oracle/runtime_distance_to_maneuver_m=4.0526
lane_model_source=trilitenet_segmentation
marker_distance_source=yolo_bbox_pinhole
```
