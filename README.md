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

## 现在从 GitHub 能复现什么

GitHub 仓库只包含源码、脚本和文档，不包含本地大资产。

被刻意排除的内容包括：

```text
.venv310/       Python 虚拟环境
models/         YOLO / TriLiteNet 权重
third_party/    CARLA / TriLiteNet / ManiSkill3 源码或二进制
runs/           视频、trace、summary 等实验产物
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
| ManiSkill valve backend smoke | 可选 CARLA | ManiSkill 环境 | 否 | 需要额外环境 |

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

python3.10 -m venv .venv310
source .venv310/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-minimal.txt
```

如果机器没有 `python3.10`，可以先用系统里已有的 Python 3.10 创建环境。CARLA 0.9.15 路径建议固定 Python 3.10。

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
MPLCONFIGDIR=/tmp/matplotlib .venv310/bin/python \
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
source .venv310/bin/activate
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

可选但用于 valve backend / 更多实验：

```text
third_party/maniskill3
models/trilitenet/base.pth
models/trilitenet/nano.pth
```

如果你是在复现作者本机实验，最稳妥的方法是从旧机器复制：

```bash
mkdir -p models third_party

rsync -a <old-machine>:/home/xingshu/workspaces/fys/stage1_closed_loop_driving/models/ models/
rsync -a <old-machine>:/home/xingshu/workspaces/fys/stage1_closed_loop_driving/third_party/carla/ third_party/carla/
rsync -a <old-machine>:/home/xingshu/workspaces/fys/stage1_closed_loop_driving/third_party/trilitenet/ third_party/trilitenet/
rsync -a <old-machine>:/home/xingshu/workspaces/fys/stage1_closed_loop_driving/third_party/maniskill3/ third_party/maniskill3/
```

如果不能复制 CARLA，可以用下载脚本：

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
MPLCONFIGDIR=/tmp/matplotlib .venv310/bin/python \
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
