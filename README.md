# LLM Steering Skill

这个仓库是“机器人开车”方向的 Stage 1 原型。当前目标不是训练机器人手臂，而是先把下面这条闭环跑通：

```text
CARLA 前视 RGB
  -> TriLiteNet 车道/可行驶区域感知
  -> YOLO 视觉 marker 距离估计
  -> DrivingState
  -> LLM 风格的受限 Python 策略代码
  -> execute_primitive(...)
  -> SteeringSkill
  -> CARLA 车辆运动
  -> 自动 ExecutionFeedback retry
```

核心结论：策略代码不直接控制车辆底层连续转向，而是只调用可替换的方向盘 primitive。后续 Stage 2 可以把当前的方向盘执行器替换成机器人/阀门/方向盘 skill，而不需要改 LLM policy code 的接口。

## 当前进展

当前最新成功的 CARLA 模型感知闭环 smoke：

```text
runs/model_perception_smoke/right_angle_left_deterministic_trilitenet_lane_yolo_marker_semantic_fixed_v3
```

结果：

```text
perception_source=trilitenet_lane_yolo_marker
state_source=carla_front_rgb_trilitenet_lane_yolo_marker_distance
final_success_rate=1.0
final_feedback_counts={"none": 1}
mean_final_max_lane_center_offset_m=1.4899
```

trace 里可以确认 LLM 可见的状态来自模型感知链路：

```text
llm_perception_source=trilitenet_lane_yolo_marker
lane_model_source=trilitenet_segmentation
marker_distance_source=yolo_bbox_pinhole
```

## 目录结构

```text
envs/             仿真器 adapter，包括 CARLA
perception/       DrivingState schema 和感知 adapter
llm_policy/       prompt、LLM client、代码校验和 runtime
skills/           SteeringSkill 接口和 oracle/simulated steering skill
robot_steering/   可替换的机器人/阀门 steering backend
feedback/         trace 到 ExecutionFeedback 的转换
experiments/      可运行实验和 smoke 脚本
scripts/          shell 入口
docs/             更详细的状态、设置和实验记录
models/           本地模型权重，不进 git
third_party/      CARLA、ManiSkill3、TriLiteNet，不进 git
runs/             实验产物，不进 git，除了 runs/README.md
```

`runs/`、`models/`、`third_party/` 和虚拟环境都是本地机器状态。GitHub 仓库只保存源码、文档和脚本，不保存视频、CARLA 包、模型权重或 venv。

## 跨机器配置

GitHub 仓库不包含运行 demo 所需的大文件：

```text
.venv310/       Python 环境，已 ignore
models/         YOLO/TriLiteNet 权重，已 ignore
third_party/    CARLA/ManiSkill3/TriLiteNet，已 ignore
runs/           视频、trace、metrics，除 runs/README.md 外已 ignore
```

新机器先 clone：

```bash
git clone git@github.com:programessi/llm-steering-skill.git
cd llm-steering-skill
```

创建 Python 3.10 环境：

```bash
python3.10 -m venv .venv310
source .venv310/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-stage1.txt
```

如果新机器的 CUDA/PyTorch 版本有特殊要求，先按 pytorch.org 给出的命令安装匹配的 `torch`/`torchvision`，然后再执行 `requirements-stage1.txt`。

恢复本地大文件。最稳妥的方法是从当前这台机器复制：

```bash
mkdir -p models third_party

rsync -a <old-machine>:/home/xingshu/workspaces/fys/stage1_closed_loop_driving/models/ models/
rsync -a <old-machine>:/home/xingshu/workspaces/fys/stage1_closed_loop_driving/third_party/carla/ third_party/carla/
rsync -a <old-machine>:/home/xingshu/workspaces/fys/stage1_closed_loop_driving/third_party/trilitenet/ third_party/trilitenet/
rsync -a <old-machine>:/home/xingshu/workspaces/fys/stage1_closed_loop_driving/third_party/maniskill3/ third_party/maniskill3/
```

恢复后至少需要这些文件：

```text
models/yolo11n.pt
models/trilitenet/small.pth
third_party/carla/CarlaUE4.sh
third_party/trilitenet/lib/models/TriLiteNet.py
```

可选但建议保留：

```text
models/trilitenet/base.pth
models/trilitenet/nano.pth
third_party/maniskill3
```

如果不能从旧机器复制 CARLA，可以用下载脚本：

```bash
scripts/download_carla_server_parallel.sh
```

如果不能复制 TriLiteNet 源码，可以重新 clone：

```bash
mkdir -p third_party
git clone https://github.com/chequanghuy/TriLiteNet.git third_party/trilitenet
```

然后把 TriLiteNet 权重放到：

```text
models/trilitenet/
```

如果要使用真实 LLM 生成 policy code，需要配置 OpenAI-compatible endpoint：

```bash
export STAGE1_LLM_BASE_URL="https://your-openai-compatible-endpoint/v1"
export STAGE1_LLM_API_KEY="..."
export STAGE1_LLM_MODEL="..."
```

确定性 smoke 不需要 LLM key。只有 `--policy-generator llm` 路径需要。

新机器建议按这个顺序验证：

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv310/bin/python \
  experiments/run_trilitenet_lane_smoke.py \
  --out runs/model_perception_smoke/trilitenet_lane_offline_smoke

CARLA_QUALITY=Low CARLA_RENDER_OFFSCREEN=0 CARLA_PORT=2000 \
  scripts/start_carla_server.sh -stdout -FullStdOutLogOutput

scripts/check_carla_rpc.sh
```

以上都通过后，再运行下面的 canonical closed-loop smoke。

## 主要运行命令

进入工程目录：

```bash
cd /home/xingshu/workspaces/fys/stage1_closed_loop_driving
```

另开一个终端启动 CARLA：

```bash
CARLA_QUALITY=Low CARLA_RENDER_OFFSCREEN=0 CARLA_PORT=2000 \
  scripts/start_carla_server.sh -stdout -FullStdOutLogOutput
```

跑实验前先检查 CARLA RPC：

```bash
scripts/check_carla_rpc.sh
```

期望输出形态：

```json
{
  "ok": true,
  "server_version": "0.9.15",
  "client_version": "0.9.15"
}
```

运行当前 canonical 闭环 smoke：

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

离线 TriLiteNet 车道感知 smoke：

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv310/bin/python \
  experiments/run_trilitenet_lane_smoke.py \
  --out runs/model_perception_smoke/trilitenet_lane_offline_smoke
```

机器人/阀门 steering backend smoke：

```bash
scripts/run_carla_stage1_auto_feedback_repeats.sh \
  --task right_angle_left \
  --trials 1 \
  --policy-generator deterministic \
  --steering-skill simulated_robot \
  --robot-skill-backend maniskill_valve
```

## 模型和第三方资产

本地应有资产：

```text
models/yolo11n.pt
models/trilitenet/small.pth
third_party/carla
third_party/maniskill3
third_party/trilitenet
```

TriLiteNet 源码在 `third_party/trilitenet`，权重在 `models/trilitenet`。CARLA 体积很大，只作为本地资产保存。

## 注意事项

CARLA Python client 需要本地 socket 权限。在 Codex 或 sandboxed 环境里，CARLA server 和 Python client 都要在能访问 `127.0.0.1:2000` 的上下文中运行；否则即使 server 正常，`client.get_world()` 也可能 timeout。

CARLA 退出时可能打印 UE4 `Signal 11`。如果视频、trace、summary 已正常生成，这个退出阶段报错不代表实验失败。

详细状态和历史记录见：

```text
docs/stage1_status.md
docs/experiment_inventory.md
docs/perception_adapter_notes.md
docs/carla_setup.md
docs/maniskill_valve_backend.md
docs/maintenance.md
```
