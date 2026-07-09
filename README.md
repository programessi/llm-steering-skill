# Stage 1 Closed-Loop Driving

This project is the stage-1 prototype for robot-driving-car work.

Current verified chain:

```text
CARLA front RGB
  -> TriLiteNet lane/drivable-area perception
  -> YOLO visual marker distance
  -> DrivingState
  -> LLM-style restricted policy code
  -> execute_primitive(...)
  -> SteeringSkill
  -> CARLA vehicle
  -> automatic ExecutionFeedback retry
```

The important result is that policy code controls the car only through
replaceable steering primitives. Stage 2 can replace the current steering
executor with a robot/valve/wheel skill without changing the policy-code
contract.

## Current Status

Latest successful CARLA model-perception closed-loop smoke:

```text
runs/model_perception_smoke/right_angle_left_deterministic_trilitenet_lane_yolo_marker_semantic_fixed_v3
```

Result:

```text
perception_source=trilitenet_lane_yolo_marker
state_source=carla_front_rgb_trilitenet_lane_yolo_marker_distance
final_success_rate=1.0
final_feedback_counts={"none": 1}
mean_final_max_lane_center_offset_m=1.4899
```

Trace confirms the LLM-visible state came from the model-backed path:

```text
llm_perception_source=trilitenet_lane_yolo_marker
lane_model_source=trilitenet_segmentation
marker_distance_source=yolo_bbox_pinhole
```

## Layout

```text
envs/             simulator adapters, including CARLA
perception/       DrivingState schema and perception adapters
llm_policy/       prompt, LLM client, generated-code validation/runtime helpers
skills/           SteeringSkill interface and oracle/simulated steering skills
robot_steering/   replaceable robot/valve steering-skill backends
feedback/         trace-to-ExecutionFeedback adapter
experiments/      runnable experiment and smoke scripts
scripts/          shell entrypoints
docs/             longer notes, setup, experiment inventory
models/           local model weights
third_party/      CARLA, ManiSkill3, TriLiteNet source/checkouts
runs/             generated experiment artifacts
```

`runs/`, `models/`, `third_party/`, and virtual environments are local machine
state. Keep source/docs/scripts under version control; do not commit generated
videos, CARLA archives, venvs, or downloaded model binaries.

## Cross-Machine Setup

The GitHub repository should contain only source code, docs, and scripts. It
does not contain the local runtime state needed to run the demos:

```text
.venv310/       Python environment, ignored
models/         YOLO/TriLiteNet weights, ignored
third_party/    CARLA/ManiSkill3/TriLiteNet checkouts, ignored
runs/           generated videos/traces/metrics, ignored except runs/README.md
```

On a new machine, clone the repo:

```bash
git clone git@github.com:<your-user>/llm-steering-skill.git
cd llm-steering-skill
```

Create the Python environment:

```bash
python3.10 -m venv .venv310
source .venv310/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-stage1.txt
```

If PyTorch wheels need a specific CUDA index on the new machine, install
`torch`/`torchvision` first with the matching command from pytorch.org, then run
the requirements install again.

Restore local assets. The most reliable path is to copy them from this working
machine:

```bash
mkdir -p models third_party

rsync -a <old-machine>:/home/xingshu/workspaces/fys/stage1_closed_loop_driving/models/ models/
rsync -a <old-machine>:/home/xingshu/workspaces/fys/stage1_closed_loop_driving/third_party/carla/ third_party/carla/
rsync -a <old-machine>:/home/xingshu/workspaces/fys/stage1_closed_loop_driving/third_party/trilitenet/ third_party/trilitenet/
rsync -a <old-machine>:/home/xingshu/workspaces/fys/stage1_closed_loop_driving/third_party/maniskill3/ third_party/maniskill3/
```

Required assets after restore:

```text
models/yolo11n.pt
models/trilitenet/small.pth
third_party/carla/CarlaUE4.sh
third_party/trilitenet/lib/models/TriLiteNet.py
```

Optional but useful:

```text
models/trilitenet/base.pth
models/trilitenet/nano.pth
third_party/maniskill3
```

If you cannot copy CARLA from another machine, use the local download script:

```bash
scripts/download_carla_server_parallel.sh
```

If you cannot copy TriLiteNet source, clone it:

```bash
mkdir -p third_party
git clone https://github.com/chequanghuy/TriLiteNet.git third_party/trilitenet
```

Then place the TriLiteNet weights under `models/trilitenet/`.

For LLM policy generation, configure an OpenAI-compatible endpoint:

```bash
export STAGE1_LLM_BASE_URL="https://your-openai-compatible-endpoint/v1"
export STAGE1_LLM_API_KEY="..."
export STAGE1_LLM_MODEL="..."
```

The deterministic smoke does not need an LLM key. The `--policy-generator llm`
path does.

After setup, validate in this order:

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv310/bin/python \
  experiments/run_trilitenet_lane_smoke.py \
  --out runs/model_perception_smoke/trilitenet_lane_offline_smoke

CARLA_QUALITY=Low CARLA_RENDER_OFFSCREEN=0 CARLA_PORT=2000 \
  scripts/start_carla_server.sh -stdout -FullStdOutLogOutput

scripts/check_carla_rpc.sh
```

Once those pass, run the canonical closed-loop smoke from the next section.

## Main Commands

Use the project Python 3.10 environment:

```bash
cd /home/xingshu/workspaces/fys/stage1_closed_loop_driving
```

Start CARLA in a separate terminal:

```bash
CARLA_QUALITY=Low CARLA_RENDER_OFFSCREEN=0 CARLA_PORT=2000 \
  scripts/start_carla_server.sh -stdout -FullStdOutLogOutput
```

Check CARLA RPC before running experiments:

```bash
scripts/check_carla_rpc.sh
```

Expected shape:

```json
{
  "ok": true,
  "server_version": "0.9.15",
  "client_version": "0.9.15"
}
```

Run the current canonical closed-loop smoke:

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
  --timeout-s 120 \
  --out runs/model_perception_smoke/right_angle_left_deterministic_trilitenet_lane_yolo_marker_semantic_fixed_v3_repro
```

Offline TriLiteNet lane perception smoke:

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv310/bin/python \
  experiments/run_trilitenet_lane_smoke.py \
  --out runs/model_perception_smoke/trilitenet_lane_offline_smoke
```

Robot/valve steering backend smoke:

```bash
scripts/run_carla_stage1_auto_feedback_repeats.sh \
  --task right_angle_left \
  --trials 1 \
  --policy-generator deterministic \
  --steering-skill simulated_robot \
  --robot-skill-backend maniskill_valve
```

## Model And Third-Party Assets

Expected local assets:

```text
models/yolo11n.pt
models/trilitenet/small.pth
third_party/carla
third_party/maniskill3
third_party/trilitenet
```

TriLiteNet source is under `third_party/trilitenet`; weights are under
`models/trilitenet`. CARLA is large and should remain local.

## Notes

CARLA Python clients need local socket permission. In Codex/sandboxed runs,
start both the CARLA server and the Python client in a context that can open
`127.0.0.1:2000`; otherwise `client.get_world()` can time out even when the
server is healthy.

CARLA may print UE4 `Signal 11` during shutdown. That has been observed after
successful runs and is not by itself evidence that the experiment failed.

Detailed status and history:

```text
docs/stage1_status.md
docs/experiment_inventory.md
docs/perception_adapter_notes.md
docs/carla_setup.md
docs/maniskill_valve_backend.md
```
