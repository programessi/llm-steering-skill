# CARLA Rendered Demo Setup

This path is for a presentable Stage 1 driving demo:

```text
CARLA front RGB camera
  -> CARLA/map oracle DrivingState
  -> restricted LLM-style policy code
  -> discrete SteeringSkill primitives
  -> CARLA ego vehicle
  -> video, trace, metrics
```

The current CARLA demo does not use a learned perception model yet. The video is
real CARLA front RGB, but `DrivingState` is still filled from CARLA map/actor
state. This is intentional for the first CARLA milestone: it isolates the
policy/skill loop before replacing individual state fields with model outputs.

## Requirements

The CARLA Python client is installed in the project-local Python 3.10 venv:

```bash
/home/xingshu/workspaces/fys/stage1_closed_loop_driving/.venv310/bin/python - <<'PY'
import carla
print(carla)
PY
```

The full CARLA server package is large, about 8 GB. The project keeps it under:

```text
/home/xingshu/workspaces/fys/stage1_closed_loop_driving/third_party/carla
```

Download and extract it with:

```bash
cd /home/xingshu/workspaces/fys
stage1_closed_loop_driving/scripts/download_carla_server.sh
```

For unstable links, the project also has a strict segmented downloader. It
keeps already complete byte ranges and re-downloads only incomplete ranges:

```bash
cd /home/xingshu/workspaces/fys
stage1_closed_loop_driving/scripts/download_carla_server_parallel.sh
```

If the official URL is slow, provide a mirror or local file URL:

```bash
CARLA_SERVER_URL=https://your.mirror/CARLA_0.9.15.tar.gz \
  stage1_closed_loop_driving/scripts/download_carla_server.sh
```

Start the CARLA simulator in another terminal:

```bash
cd /home/xingshu/workspaces/fys
stage1_closed_loop_driving/scripts/start_carla_server.sh
```

The local default starts a visible X11 window with low quality and no sound,
because that is the path verified on this machine. To force offscreen rendering:

```bash
CARLA_RENDER_OFFSCREEN=1 stage1_closed_loop_driving/scripts/start_carla_server.sh
```

## Run

From the workspace root:

```bash
cd /home/xingshu/workspaces/fys
stage1_closed_loop_driving/scripts/run_carla_rendered_demo.sh
```

For the fixed-point driving-test primitive demos:

```bash
cd /home/xingshu/workspaces/fys
stage1_closed_loop_driving/scripts/run_carla_driving_test_demos.sh
```

The explicit equivalent is:

```bash
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
  --fps 20 \
  --timeout-s 60
```

Outputs:

```text
stage1_closed_loop_driving/runs/carla_rendered_demo/front_rgb.mp4
stage1_closed_loop_driving/runs/carla_rendered_demo/preview.png
stage1_closed_loop_driving/runs/carla_rendered_demo/trace.csv
stage1_closed_loop_driving/runs/carla_rendered_demo/summary.json
stage1_closed_loop_driving/runs/carla_rendered_demo/policy.py
```

Driving-test outputs:

```text
stage1_closed_loop_driving/runs/carla_driving_test_demos/right_angle_left/front_rgb.mp4
stage1_closed_loop_driving/runs/carla_driving_test_demos/lane_change_left/front_rgb.mp4
stage1_closed_loop_driving/runs/carla_driving_test_demos/pull_over_right/front_rgb.mp4
stage1_closed_loop_driving/runs/carla_driving_test_demos/summary.json
```

## Current Boundary

The summary explicitly records:

```text
camera_frame: carla_front_rgb
state_source: carla_map_oracle
policy_mode: llm_feedback
```

The trace records per-step primitive execution:

```text
skill_name
steering_primitive
target_angle_rad
speed_mps
steering_angle_rad
lane_center_offset_m
heading_error_rad
lane_curvature
front_vehicle_distance_m
```

## Next Replacement

Once this demo runs on a machine with CARLA installed, replace oracle fields in
this order:

```text
front_vehicle_exists/front_vehicle_distance_m -> YOLO + CARLA depth or monocular depth
lane_center_offset_m/heading_error_rad -> lane/drivable-area model postprocess
lane_curvature -> route/map prior first, learned estimator later
```

The policy code and `SteeringSkill` interface should not change during those
replacements.

## Current Local Status

Verified locally:

```text
CARLA Python client: installed in .venv310
import carla: works in .venv310
CARLA server archive: downloaded and validated
CARLA server install: extracted under third_party/carla
CARLA server binary: third_party/carla/CarlaUE4.sh exists and is executable
CARLA Python RPC: get_server_version returns 0.9.15
CARLA rendered demo: completed
```

The archive size validated against the official CDN size:

```text
CARLA_0.9.15.tar.gz: 8386636048 bytes
tar -tzf: passed
```

Successful local server command shape:

```bash
env VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  __GLX_VENDOR_LIBRARY_NAME=nvidia \
  SDL_VIDEODRIVER=x11 \
  stage1_closed_loop_driving/third_party/carla/CarlaUE4.sh \
  -quality-level=Low -nosound -carla-rpc-port=2000
```

Successful local demo result:

```text
output: stage1_closed_loop_driving/runs/carla_rendered_demo
video: front_rgb.mp4, 1280x720, 20 FPS, 241 frames
success: true
lane_departure: false
mean_lane_center_offset_m: 0.2848185018194218
max_lane_center_offset_m: 1.1864491704063767
state_source: carla_map_oracle
camera_frame: carla_front_rgb
primitive_call_count: 45
```

Successful local driving-test primitive result:

```text
output: stage1_closed_loop_driving/runs/carla_driving_test_demos
demo_count: 3
success_count: 3
state_source: carla_map_oracle
camera_frame: carla_front_rgb
steering execution: ramped over each primitive duration
speed range: roughly 2.2-4.0 m/s

right_angle_left:
  hold_center -> hard_left -> hold_left -> return_center -> hold_center
lane_change_left:
  hold_center -> medium_left -> medium_right -> return_center -> hold_center
pull_over_right:
  hold_center -> medium_right -> medium_left -> return_center -> hold_center_stop
```

Important execution note for Codex/sandboxed runs: local CARLA RPC access to
`127.0.0.1:2000` must run in a context that can open local sockets. In this
session, direct sandboxed socket creation was blocked, while the same Python
client worked when run with local-network permission.

Current RPC healthcheck:

```bash
stage1_closed_loop_driving/scripts/check_carla_rpc.sh
```

Expected output shape:

```json
{
  "ok": true,
  "server_version": "0.9.15",
  "world": "Carla/Maps/Town10HD_Opt"
}
```

If Codex runs the client from a restricted sandbox, the symptom can look like a
CARLA server timeout even when the server is healthy:

```text
RuntimeError: time-out while waiting for the simulator
```

In that case, rerun the client command in a local-socket-enabled/elevated
context. The server and client are both required: starting only the UE process
with elevated permissions is not enough if the Python client remains sandboxed.

Debug server startup with stdout logs:

```bash
CARLA_QUALITY=Low CARLA_RENDER_OFFSCREEN=0 CARLA_PORT=2000 \
  stage1_closed_loop_driving/scripts/start_carla_server.sh \
  -stdout -FullStdOutLogOutput
```
