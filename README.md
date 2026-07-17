# LLM Steering Skill

这是一个“机器人开车”方向的 Stage 1 原型。它验证的核心不是端到端训练驾驶模型，而是验证下面这个软件边界：

```text
DrivingState
  -> LLM / LLM-style restricted Python policy
  -> execute_primitive(...)
  -> replaceable SteeringSkill
  -> simulator / robot steering backend
  -> ExecutionFeedback
  -> retry / correction
```

策略代码不能直接控制车辆底层连续转向，只能调用受限的 steering primitive。这样后续可以把当前的 CARLA oracle steering executor 替换成机器人、阀门、方向盘 skill，而不需要改 LLM policy 的接口。

## 0. 直接复制粘贴：安装和验证

下面先给命令。后面的章节解释每条命令在复现什么。

### 0.1 最小安装：不装 CARLA、不装模型、不需要 LLM key

这个流程用于确认仓库源码能跑通。

```bash
git clone https://github.com/programessi/llm-steering-skill.git
cd llm-steering-skill

python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-minimal.txt

MPLCONFIGDIR=/tmp/matplotlib scripts/run_robot_steering_skill_bench.sh \
  --out runs/robot_steering_skill_bench_smoke

MPLCONFIGDIR=/tmp/matplotlib python experiments/run_stage1_demo.py \
  --task lane_keeping \
  --state-source perceived \
  --policy-mode llm_feedback \
  --out runs/stage1_demo_smoke
```

成功时应看到：

```text
robot steering bench: success_rate=1.0
stage1 demo: success=true
```

项目脚本会自动找 Python，优先级是：

```text
PYTHON 环境变量
-> .venv310/bin/python
-> .venv/bin/python
-> python
```

所以你可以用 `.venv`，也可以用 `.venv310`。如果你想显式指定：

```bash
PYTHON="$PWD/.venv/bin/python" scripts/run_robot_steering_skill_bench.sh
```

### 0.2 全量 Python 依赖：准备跑 CARLA / YOLO / TriLiteNet / ManiSkill

如果要跑完整实验，在同一个 venv 里继续安装全量依赖：

```bash
source .venv/bin/activate
python -m pip install -r requirements-stage1.txt
```

如果你的机器 CUDA / PyTorch 版本比较特殊，先按 pytorch.org 的命令安装匹配的 `torch` / `torchvision`，再执行：

```bash
python -m pip install -r requirements-stage1.txt
```

检查关键包是否能 import：

```bash
python - <<'PY'
mods = ["numpy", "cv2", "matplotlib", "requests", "torch", "torchvision", "ultralytics", "carla", "gymnasium", "mani_skill", "sapien"]
for name in mods:
    try:
        mod = __import__(name)
        print(name, getattr(mod, "__version__", "ok"))
    except Exception as exc:
        print(name, "FAILED", type(exc).__name__, exc)
PY
```

### 0.3 安装第三方源码 submodules

这个仓库用 git submodule 管理两个源码依赖：

```text
third_party/trilitenet   -> https://github.com/chequanghuy/TriLiteNet.git
third_party/maniskill3   -> https://github.com/haosulab/ManiSkill.git
```

推荐 clone 时直接拉 submodules：

```bash
git clone --recurse-submodules https://github.com/programessi/llm-steering-skill.git
cd llm-steering-skill
```

如果你已经普通 clone 了仓库：

```bash
git submodule update --init --recursive
```

检查 submodule commit：

```bash
git submodule status
```

当前固定的本地验证版本是：

```text
TriLiteNet: 4ac49477ff73dc9f45497ad4c397e92e6089436b
ManiSkill:  42b68244c1497cef889b04c4f4a78aa01c927f4e
```

CARLA 不做 submodule。它是约 8GB 的二进制 release 包，继续用下载脚本放到：

```text
third_party/carla
```

### 0.4 安装 CARLA server

CARLA Python client 来自 pip 包，CARLA server 是单独的大二进制，约 8 GB。

```bash
mkdir -p third_party
scripts/download_carla_server_parallel.sh
chmod +x third_party/carla/CarlaUE4.sh
```

启动 CARLA server：

```bash
CARLA_QUALITY=Low CARLA_RENDER_OFFSCREEN=0 CARLA_PORT=2000 \
  scripts/start_carla_server.sh -stdout -FullStdOutLogOutput
```

另开一个终端检查 RPC：

```bash
cd llm-steering-skill
source .venv/bin/activate
scripts/check_carla_rpc.sh
```

期望输出：

```json
{
  "ok": true,
  "server_version": "0.9.15"
}
```

如果 CARLA 官方下载慢，可以换镜像：

```bash
CARLA_SERVER_URL=https://your.mirror/CARLA_0.9.15.tar.gz \
  scripts/download_carla_server_parallel.sh
```

如果你已经手动下载并解压了 CARLA，也可以直接指定路径：

```bash
export CARLA_ROOT=/path/to/CARLA_0.9.15
CARLA_QUALITY=Low CARLA_PORT=2000 scripts/start_carla_server.sh
```

### 0.5 安装视觉模型资产

YOLO11n 权重可以让 Ultralytics 自动下载，然后放到项目约定路径：

```bash
source .venv/bin/activate
mkdir -p models
python - <<'PY'
from ultralytics import YOLO
YOLO("yolo11n.pt")
PY
mv yolo11n.pt models/yolo11n.pt
```

TriLiteNet 源码由 submodule 提供。如果目录不存在，执行：

```bash
git submodule update --init --recursive third_party/trilitenet
```

TriLiteNet 权重目前没有随本仓库公开托管，需要你从已有机器或权重来源复制：

```bash
mkdir -p models/trilitenet
cp /path/to/small.pth models/trilitenet/small.pth
```

如果你从作者旧机器复制完整资产，模型权重和 CARLA 可以复制；submodule 源码建议仍用 `git submodule update --init --recursive`：

```bash
mkdir -p models third_party
rsync -a <old-machine>:/home/xingshu/workspaces/fys/stage1_closed_loop_driving/models/ models/
rsync -a <old-machine>:/home/xingshu/workspaces/fys/stage1_closed_loop_driving/third_party/carla/ third_party/carla/
git submodule update --init --recursive
```

### 0.6 安装 / 验证 ManiSkill valve backend

如果只用已发布包，`requirements-stage1.txt` 已经安装：

```text
mani_skill==3.0.1
sapien==3.0.3
```

验证 Python 包能 import：

```bash
source .venv/bin/activate
python - <<'PY'
import gymnasium as gym
import mani_skill.envs  # noqa: F401
import sapien
print("mani_skill import ok")
print("sapien", getattr(sapien, "__version__", "ok"))
print("RotateValveLevel0-v1 registered:", "RotateValveLevel0-v1" in gym.registry)
PY
```

如果你需要和作者本地完全一致的 ManiSkill 源码，使用 submodule：

```bash
git submodule update --init --recursive third_party/maniskill3
python -m pip install -e third_party/maniskill3
```

验证 RotateValve 环境可以创建并读 valve qpos：

```bash
python - <<'PY'
import gymnasium as gym
import mani_skill.envs  # noqa: F401

env = gym.make("RotateValveLevel0-v1", num_envs=1, obs_mode="state", render_mode=None)
env.reset(seed=0)
print("has_valve", hasattr(env.unwrapped, "valve"))
print("valve_qpos", env.unwrapped.valve.qpos.detach().cpu().numpy().tolist())
env.close()
PY
```

注意：ManiSkill / Sapien 环境创建需要可用的 GPU/Vulkan/render device 权限。在无 GPU、无 Vulkan 或受限 sandbox 里，可能出现类似错误：

```text
RuntimeError: Failed to find a supported physical device
```

这表示设备/渲染后端不可用，不是本仓库 Python import 路径的问题。换到有 GPU/Vulkan/Sapien 权限的本机终端运行，或先使用不依赖 ManiSkill 的 `kinematic_valve` backend。

### 0.7 跑 CARLA oracle smoke

这个不需要 YOLO/TriLiteNet，只需要 CARLA server 正在运行：

```bash
MPLCONFIGDIR=/tmp/matplotlib scripts/run_carla_rendered_demo.sh \
  --out runs/carla_rendered_demo_repro
```

### 0.8 跑完整 CARLA + TriLiteNet + YOLO smoke

这个需要 CARLA server、YOLO 权重、TriLiteNet 源码和 TriLiteNet 权重：

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
  --out runs/model_perception_smoke/right_angle_left_deterministic_trilitenet_lane_yolo_marker_repro
```

## 现在从 GitHub 能复现什么

GitHub 仓库只包含源码、脚本和文档，不包含本地大资产。

被刻意排除的内容包括：

```text
.venv/ 或 .venv310/      Python 虚拟环境
models/                    YOLO / TriLiteNet 权重
third_party/carla/          CARLA 二进制 server
third_party/downloads/      CARLA 下载缓存
runs/                      视频、trace、summary 等实验产物
```

因此复现分成几个层级：

| 复现目标 | 需要 CARLA server | 需要模型权重 | 需要 LLM API key | GitHub clone 后能否直接复现 |
|---|---:|---:|---:|---:|
| robot steering skill bench | 否 | 否 | 否 | 是 |
| kinematic Stage-1 baseline | 否 | 否 | 否 | 是 |
| LLM policy generation smoke | 否 | 否 | 是 | 配 key 后可以 |
| TriLiteNet lane smoke | 否 | 是 | 否 | 需要额外资产 |
| CARLA oracle steering smoke | 是 | 否 | 否 | 需要额外资产 |
| CARLA + TriLiteNet + YOLO canonical smoke | 是 | 是 | 否 | 需要额外资产 |
| ManiSkill valve backend smoke | 通常需要 CARLA | ManiSkill/Sapien render device | 否 | 需要额外环境 |

如果只是检查仓库是否安装正确，先跑“GitHub-only 最小复现”。不要一开始就跑完整 CARLA + 模型感知闭环。

## 目录结构

```text
envs/             simulator adapters，包括 kinematic fallback、CARLA、MetaDrive
perception/       DrivingState schema 和感知 adapter
llm_policy/       prompt、LLM client、代码校验和 restricted runtime
skills/           SteeringSkill 接口和 oracle / simulated robot steering skill
robot_steering/   可替换的机器人/阀门 steering backend
feedback/         trace 到 ExecutionFeedback 的转换
experiments/      可运行实验和 smoke 脚本
scripts/          shell 入口
docs/             更详细的状态、设置和实验记录
runs/             实验输出目录，只有 runs/README.md 进 git
```

## 1. GitHub-only 最小复现

这个路径不需要 CARLA、不需要 YOLO/TriLiteNet 权重、不需要 LLM key。它验证源码、restricted policy runtime、kinematic simulator、SteeringSkill 接口和 robot steering skill 数据格式。

### 1.1 创建最小 Python 环境

建议 Python 3.10。

```bash
git clone https://github.com/programessi/llm-steering-skill.git
cd llm-steering-skill

python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-minimal.txt
```

如果机器没有 `python3.10`，可以先用系统里已有的 Python 3.10 创建环境。CARLA 0.9.15 路径建议固定 Python 3.10。项目脚本同时兼容 `.venv` 和 `.venv310`。

### 1.2 跑 robot steering skill bench

```bash
MPLCONFIGDIR=/tmp/matplotlib scripts/run_robot_steering_skill_bench.sh \
  --out runs/robot_steering_skill_bench_smoke
```

期望输出里应包含类似字段：

```text
skill_impl=StubRobotSteeringSkill
success_rate=1.0
results_csv=runs/robot_steering_skill_bench_smoke/results.csv
episodes_dir=runs/robot_steering_skill_bench_smoke/episodes
```

这个 smoke 验证的是 Stage-2 robot steering skill 的数据接口：

```text
SteeringSkillCommand
  -> RobotSteeringSkill.execute(...)
  -> SteeringTrajectorySample
  -> observations / actions / success / final_angle_error_rad
```

### 1.3 跑 kinematic Stage-1 baseline

```bash
MPLCONFIGDIR=/tmp/matplotlib python \
  experiments/run_stage1_baselines.py \
  --out runs/stage1_baselines_smoke \
  --video-task lane_keeping
```

这个命令会在本地 kinematic simulator 里跑多组任务、状态源和 policy mode 组合，输出：

```text
runs/stage1_baselines_smoke/summary.json
runs/stage1_baselines_smoke/*/policy.py
runs/stage1_baselines_smoke/*/summary.json
runs/stage1_baselines_smoke/*/trace.csv
```

如果上面两个命令通过，说明最小源码复现成功。

## 2. 可选：LLM policy generation smoke

只有真正调用在线 LLM 生成 policy code 时才需要 API key。确定性 policy / kinematic baseline 不需要。

配置 OpenAI-compatible endpoint：

```bash
export STAGE1_LLM_BASE_URL="https://your-openai-compatible-endpoint/v1"
export STAGE1_LLM_API_KEY="..."
export STAGE1_LLM_MODEL="..."
```

运行：

```bash
MPLCONFIGDIR=/tmp/matplotlib scripts/generate_llm_policy_smoke.sh
```

这个命令只验证 LLM 生成的 Python policy 是否能通过 AST safety checker，不需要 CARLA。

## 3. 可选：全量 Python 依赖

如果要跑 CARLA、YOLO/TriLiteNet 或 ManiSkill backend，再安装全量依赖：

```bash
source .venv/bin/activate
python -m pip install -r requirements-stage1.txt
```

注意：`requirements-stage1.txt` 包含 CARLA、ManiSkill、Sapien、Torch、Ultralytics 等重依赖。不同 CUDA / driver / Python 环境下可能需要先按本机环境安装合适的 `torch` / `torchvision`，再安装其余依赖。

本地已验证过的开发环境大致为：

```text
Python 3.10
carla==0.9.15
mani_skill==3.0.1
sapien==3.0.3
torch / torchvision with local CUDA support
```

## 4. 可选：准备模型和第三方资产

完整视觉闭环需要以下本地资产：

```text
models/yolo11n.pt
models/trilitenet/small.pth
third_party/carla/CarlaUE4.sh
third_party/trilitenet/lib/models/TriLiteNet.py
```

由 submodule 提供：

```text
third_party/trilitenet
third_party/maniskill3
```

可选但用于更多 TriLiteNet 实验：

```text
models/trilitenet/base.pth
models/trilitenet/nano.pth
```

如果你是在复现作者本机实验，模型权重和 CARLA 可以从旧机器复制；源码依赖用 submodule：

```bash
mkdir -p models third_party

rsync -a <old-machine>:/home/xingshu/workspaces/fys/stage1_closed_loop_driving/models/ models/
rsync -a <old-machine>:/home/xingshu/workspaces/fys/stage1_closed_loop_driving/third_party/carla/ third_party/carla/
git submodule update --init --recursive
```

如果不能复制 CARLA，可以用下载脚本：

```bash
scripts/download_carla_server_parallel.sh
```

TriLiteNet 源码由 submodule 安装：

```bash
git submodule update --init --recursive third_party/trilitenet
```

然后把 TriLiteNet 权重放到：

```text
models/trilitenet/small.pth
```

当前仓库没有公开托管 TriLiteNet 权重和 CARLA 二进制；因此视觉闭环不是“GitHub-only”复现。

## 5. 可选：TriLiteNet lane offline smoke

这个 smoke 需要：

```text
third_party/trilitenet
models/trilitenet/small.pth
一张输入图像
```

脚本默认输入图像是：

```text
runs/model_perception_smoke/marker_frame.png
```

但 `runs/` 不进 git，所以新 clone 后需要自己提供 `--image`。

示例：

```bash
MPLCONFIGDIR=/tmp/matplotlib python \
  experiments/run_trilitenet_lane_smoke.py \
  --image /path/to/front_rgb.png \
  --out runs/model_perception_smoke/trilitenet_lane_offline_smoke \
  --trilitenet-root third_party/trilitenet \
  --trilitenet-weights models/trilitenet/small.pth
```

输出包括：

```text
summary.json
trilitenet_overlay.png
lane_mask.png
drivable_mask.png
```

## 6. 可选：CARLA oracle steering smoke

这个路径验证 CARLA 前视 RGB 视频、CARLA/map oracle DrivingState、restricted policy runtime 和 SteeringSkill，不需要 YOLO/TriLiteNet 权重。

先启动 CARLA server：

```bash
CARLA_QUALITY=Low CARLA_RENDER_OFFSCREEN=0 CARLA_PORT=2000 \
  scripts/start_carla_server.sh -stdout -FullStdOutLogOutput
```

另开终端检查 RPC：

```bash
scripts/check_carla_rpc.sh
```

期望输出形态：

```json
{
  "ok": true,
  "server_version": "0.9.15"
}
```

运行 CARLA rendered demo：

```bash
MPLCONFIGDIR=/tmp/matplotlib scripts/run_carla_rendered_demo.sh \
  --out runs/carla_rendered_demo_repro
```

输出：

```text
runs/carla_rendered_demo_repro/front_rgb.mp4
runs/carla_rendered_demo_repro/preview.png
runs/carla_rendered_demo_repro/trace.csv
runs/carla_rendered_demo_repro/summary.json
runs/carla_rendered_demo_repro/policy.py
```

## 7. 可选：CARLA + TriLiteNet + YOLO canonical smoke

这是当前最完整的 Stage-1 闭环：

```text
CARLA front RGB
  -> TriLiteNet lane / drivable-area perception
  -> YOLO visual marker distance
  -> DrivingState
  -> LLM-style restricted Python policy
  -> execute_primitive(...)
  -> SteeringSkill
  -> CARLA vehicle motion
  -> automatic ExecutionFeedback retry
```

需要提前准备：

```text
CARLA server 正在运行
models/yolo11n.pt
models/trilitenet/small.pth
third_party/trilitenet
```

运行 deterministic canonical smoke：

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
  --out runs/model_perception_smoke/right_angle_left_deterministic_trilitenet_lane_yolo_marker_repro
```

本地历史成功结果见：

```text
docs/perception_adapter_notes.md
docs/stage1_status.md
docs/experiment_inventory.md
```

## 8. 可选：robot / valve steering backend

Stage 1 的 policy 只调用 `execute_primitive(...)`。底层可以从 `OracleSteeringSkill` 切到 `SimulatedRobotSteeringSkill`。

### 8.1 Kinematic valve backend

这个后端不需要真实 ManiSkill robot policy。它把 steering command 转成 simulated valve joint trajectory，再把 measured valve angle 喂回 CARLA。

```bash
MPLCONFIGDIR=/tmp/matplotlib scripts/run_carla_stage1_auto_feedback_repeats.sh \
  --task right_angle_left \
  --trials 1 \
  --policy-generator deterministic \
  --steering-skill simulated_robot \
  --robot-skill-backend kinematic_valve \
  --out runs/valve_backend_smoke/right_angle_left_deterministic_kinematic_valve
```

### 8.2 ManiSkill valve backend

这个后端创建 ManiSkill `RotateValveLevel0-v1`，把 target steering angle 写成 valve qpos trajectory，再读取 `env.unwrapped.valve.qpos` 作为 measured wheel angle。

```bash
MPLCONFIGDIR=/tmp/matplotlib scripts/run_carla_stage1_auto_feedback_repeats.sh \
  --task right_angle_left \
  --trials 1 \
  --policy-generator deterministic \
  --steering-skill simulated_robot \
  --robot-skill-backend maniskill_valve \
  --out runs/valve_backend_smoke/right_angle_left_deterministic_maniskill_valve
```

注意：当前 ManiSkill backend 是接口桥接验证，仍然直接设置 valve qpos；它还不是 learned DClaw/Panda valve policy。

详情见：

```text
docs/maniskill_valve_backend.md
docs/robot_steering_skill_bench.md
```

## 9. 常见问题

### `runs/model_perception_smoke/marker_frame.png` 不存在

这是正常的。`runs/` 是实验产物目录，不进 git。运行 `run_trilitenet_lane_smoke.py` 时请显式传入：

```bash
--image /path/to/front_rgb.png
```

### CARLA RPC timeout

先确认 server 是否启动：

```bash
scripts/check_carla_rpc.sh
```

如果在 sandbox / Codex 环境里跑，可能 Python client 没有本地 socket 权限。症状通常是：

```text
RuntimeError: time-out while waiting for the simulator
```

这种情况下需要在允许访问 `127.0.0.1:2000` 的上下文运行 Python client。只给 CARLA server 权限不够，client 也必须能打开本地 socket。

### Matplotlib cache warning

如果看到：

```text
Matplotlib created a temporary cache directory...
```

用：

```bash
MPLCONFIGDIR=/tmp/matplotlib ...
```

### CARLA 退出时 UE4 `Signal 11`

如果视频、trace、summary 已经正常生成，CARLA 退出阶段的 UE4 `Signal 11` 不一定代表实验失败。

## 10. 相关文档

```text
docs/architecture.md
docs/carla_setup.md
docs/perception_adapter_notes.md
docs/maniskill_valve_backend.md
docs/robot_steering_skill_bench.md
docs/experiment_inventory.md
docs/stage1_status.md
docs/maintenance.md
```
