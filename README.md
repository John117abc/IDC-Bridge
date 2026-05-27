# IDC-Bridge

把"集成决策与控制"（IDC, Integrated Decision and Control）论文的算法跑在 Waymo 开放数据集 + GPUDrive 模拟器上。

使用单线程训练循环跑 200 条道路场景同时仿真的在线强化学习。

## 算法简介

IDC 的核心思想是**一边开一边学**：

- **PEV（策略评估）**：Critic 网络评估当前 state 的价值，每步更新。
- **PIM（策略改进）**：每隔 N 步，Actor 通过可微运动学模型推演未来 H 步轨迹，计算跟踪误差 + 碰撞风险的总成本，反向传播改进策略。每次改进放大惩罚系数 ρ。

Actor/Critic 为两个独立 MLP（LayerNorm + ELU），动作空间为加速度和方向盘转角。

## 和原论文的差异

1. **网络结构**：Actor/Critic 独立网络，无共享参数（原论文共享 latent embed）
2. **训练架构**：单线程主循环替代 30 个异步 learner
3. **环境**：Nocturne → GPUDrive（Waymo 数据）
4. **候选参考路径**：3 条 expert pos + 曲率限速路径（左/中/右 lateral offset），每 episode 固定选一条。道路宽度从 GPUDrive 动态读取
5. **碰撞模型**：前后双圆模型替代单圆
6. **状态空间**：59 维，含当前误差 + 3 个前瞻点（t+3/t+6/t+9）的横向/航向/道路边界预览
7. **性能优化**：批量 tensor 操作 + GPU `ref_tensor` 预索引，消除 rollout 中的 Python 循环瓶颈
8. **统一框架**：`rho=0` 即纯追踪诊断模式，`rho>0` 即含 collision/road penalty，无需额外 flag
9. **YAML 层级配置**：`base.yaml`（共享）→ `train.yaml`/`eval.yaml` 继承覆盖，CLI 仅需指定差异项
10. **条件 penalty 放大**：仅在 rollout 中检测到碰撞/越界违规时才放大 ρ，防止噪声梯度污染
11. **密度范围筛选**：离线扫描 Waymo 场景周车密度，训练/评估可按密度区间 `[min, max]` 选择世界

## 快速开始

### 安装

```bash
pip install gpudrive torch numpy matplotlib pyyaml
```

### 配置文件

配置采用 **`_base` 继承机制**，共享参数放 `base.yaml`，train/eval 各自覆盖差异：

```
src/configs/
├── base.yaml      # training/agent/dynamics/world/diag — 共享
├── train.yaml     # _base: base.yaml + env(epochs=600)/paths/viz(gif=false)
└── eval.yaml      # _base: base.yaml + env(epochs=1)/paths/viz(gif=true)
```

```bash
# 训练（自动用 train.yaml）
python idc-train.py --data-dir /path/to/data

# 评估（自动用 eval.yaml）
python idc-eval.py --data-dir /path/to/data --model-path /path/to/model.pth

# 手动指定
python idc-train.py --config src/configs/train.yaml --data-dir /path/to/data
```

CLI 参数仅覆盖 YAML 中的值，优先级：CLI > YAML > base。

### 训练

```bash
# 纯追踪诊断（rho=0，无碰撞/道路惩罚）— 用全量数据池，不限稠密场景
python src/scripts/train/idc-train.py \
    --data-dir /path/to/waymo/data \
    --config configs/default.yaml \
    --init-penalty 0 \
    --epochs 300

# 正常 penalty 训练（使用 YAML 默认 rho=1.0, pim_interval=30）
# resample 时优先从稠密场景池抽取
python src/scripts/train/idc-train.py \
    --data-dir /path/to/waymo/data

# 覆盖 YAML 参数
python src/scripts/train/idc-train.py \
    --data-dir /path/to/waymo/data \
    --num-worlds 200 \
    --lr-actor 5e-5 \
    --max-penalty 15.0

# 按密度范围选择场景（需先离线扫描）
python src/scripts/train/idc-train.py \
    --data-dir /path/to/waymo/data \
    --density-cache-file /workspace/data/density_full.json \
    --min-partner-density 5 --max-partner-density 10
```

### 评估

```bash
python src/scripts/eval/idc-eval.py \
    --data-dir /path/to/waymo/data \
    --model-path /path/to/model.pth
```

### 离线密度扫描

训练前可预先扫描所有场景的周车密度，生成 JSON 缓存供训练/评估按密度区间选择：

```bash
# 全量扫描（~18 分钟）
python src/scripts/utils/scan_density.py \
    --data-dir /path/to/tfrecords \
    --output /workspace/data/density_full.json \
    --batch-size 150

# 快速抽样（~30 秒）
python src/scripts/utils/scan_density.py \
    --data-dir /path/to/tfrecords \
    --output /workspace/data/density_sample.json \
    --batch-size 150 --sample 2000
```

输出 JSON 包含各文件的周车数量分布统计，配合 `--min-partner-density` / `--max-partner-density` 使用。

**Waymo 70k 场景实际分布：**

| 密度区间 | 文件数 | 占比 |
|----------|--------|------|
| 0 | 381 | 0.5% |
| 1–2 | 3,275 | 4.7% |
| 3–5 | 7,780 | 11.1% |
| 6–10 | 13,984 | 19.9% |
| 11–20 | 19,565 | 27.9% |
| **21–30** | 10,155 | 14.5% |
| **31+** | 15,032 | 21.4% |

均值 19.19 / 中位数 15 / 范围 0–63。约 36% 的场景密度 ≥ 21（适合 penalty 训练）。

### 关键 CLI 参数

CLI 值非空时覆盖 YAML 中的同名参数，以下为常用覆盖项：

| 参数 | 说明 |
|------|------|
| `--config` | YAML 配置文件路径（train 默认 `configs/train.yaml`，eval 默认 `configs/eval.yaml`） |
| `--data-dir` | Waymo tfrecord 数据目录（**必填**） |
| `--model-path` | 模型 checkpoint 路径（评估必填） |
| `--init-penalty` | 初始 ρ（0=纯追踪，1.0=含 penalty） |
| `--num-worlds` | 并行仿真世界数 |
| `--epochs` | 训练轮数 |
| `--lr-actor` | Actor 学习率 |
| `--lr-critic` | Critic 学习率 |
| `--max-penalty` | ρ 上限 |
| `--amplifier-c` | ρ 每次 PIM 放大倍率 |
| `--device` | 计算设备（cuda/cpu） |
| `--seed` | 随机种子 |
| `--density-cache-file` | 离线扫描的全量密度 JSON |
| `--min-partner-density` | 周车密度下限（含），0=不限 |
| `--max-partner-density` | 周车密度上限（含），默认 30 |
| `--dense-sample-size` | 候选池大小，0=不限 |

### YAML 配置速查

完整参数见 `configs/default.yaml`，关键分组：

| 分组 | 包含参数 |
|------|----------|
| `env` | `num_worlds`, `max_agents`, `epochs`, `seed`, `device` |
| `training` | `dt`, `horizon`, `batch_size`, `hidden_dim`, `lr_actor`, `lr_critic`, `pim_interval`, `gamma`, `buffer_capacity` |
| `agent` | `pos_err_weight`, `heading_err_weight`, 等 7 个 cost weight；`noise_std`/`decay_rate`/`min`；`init_penalty`/`max_penalty`/`amplifier_c`；`D_veh_safe`/`D_road_safe`/`half_length`/`half_width` |
| `dynamics` | `wheelbase`, `lr_ratio`, `v_max` |
| `world` | `min_partner_density`, `max_partner_density`, `dense_sample_size`, `max_bad_worlds`, `filter_threshold`, `dataset_size`, `density_cache_file` |
| `paths` | `file_dir`, `model_path`, `save_freq`, `load_model` |
| `diag` | `fix_speed`, `fix_heading`, `no_sign` |
| `viz` | `gif_enabled`, `gif_fps`, `gif_max_worlds`, `gif_save_dir`, `gif_world_selection`, `gif_view_mode`, `gif_zoom_radius` |

### 配置继承说明

| 文件 | 说明 |
|------|------|
| `base.yaml` | 5 个共享分组（`training`/`agent`/`dynamics`/`world`/`diag`） |
| `train.yaml` | `_base: base.yaml` + `env`(epochs=600) + `paths`(load_model=true) + `viz`(gif=false) |
| `eval.yaml` | `_base: base.yaml` + `env`(epochs=1, num_worlds=10) + `paths` + `viz`(gif=true) |
| `viz` | `gif_enabled`, `gif_fps`, `gif_max_worlds`, `gif_save_dir`, `gif_world_selection`, `gif_view_mode`, `gif_zoom_radius` |

### GIF 可视化

评估脚本支持三种视图模式，通过 `viz` 配置控制：

```bash
# bird_2d —— 全局俯视图（默认）
python idc-eval.py ... --gif-view-mode bird_2d --gif-zoom-radius 70

# bird_3d —— 3D 透视投影
python idc-eval.py ... --gif-view-mode bird_3d

# agent_pov —— 自车局部视角（以 ego 为中心，obs_radius 范围）
python idc-eval.py ... --gif-view-mode agent_pov

# 关闭 GIF 加速评估
python idc-eval.py ... --gif-enabled false

# 随机录 5 个 world
python idc-eval.py ... --gif-max-worlds 5 --gif-world-selection random
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `gif_enabled` | `true` | 是否生成 GIF（评估时关闭可提速） |
| `gif_fps` | `5` | GIF 帧率 |
| `gif_max_worlds` | `10` | 最多录制 world 数（0=全部） |
| `gif_save_dir` | `/workspace/data/gifs` | GIF 输出目录 |
| `gif_world_selection` | `first` | 录制 world 选择策略：`first`（前 N）/ `random`（随机 N） |
| `gif_view_mode` | `bird_2d` | 视图模式：`bird_2d` / `bird_3d` / `agent_pov` |
| `gif_zoom_radius` | `70` | 鸟瞰视图缩放半径

## State 布局（59 维）

| 区间 | 维度 | 内容 |
|------|------|------|
| [0:6] | 6 | 自车: x, y, v_lon, v_lat, phi, omega |
| [6:38] | 32 | 周车: 8 车 × (x, y, phi, v) |
| [38:46] | 8 | validity mask |
| [46:58] | 12 | ref_error: dp/dphi/dv + 3×(lat+dphi+road_dist) @ t+3/6/9 |
| [58:59] | 1 | temporal index |

## 文件结构

```
.
├── configs/
│   ├── base.yaml              # 共享参数
│   ├── train.yaml             # 训练覆盖
│   └── eval.yaml              # 评估覆盖
├── src/
│   ├── agents/idc_agent.py
│   ├── env/
│   │   ├── env_utils.py
│   │   ├── world_manager.py
│   │   └── idc_state_builder.py
│   ├── models/
│   │   ├── kinematic_bicycle.py
│   │   └── continuous_actor_critic.py
│   ├── buffer/per_buffer.py
│   ├── scripts/
│   │   ├── train/idc-train.py
│   │   ├── eval/idc-eval.py
│   │   └── utils/scan_density.py   # 离线密度扫描
│   └── utils/
│       ├── config.py               # YAML _base 继承 + CLI 合并
│       ├── traj_visualizer.py
│       └── ...
├── info.md
└── README.md
├── info.md                           # 完整开发记录
├── README.md
└── requirements.txt
```

## 踩过的坑

完整记录见 `info.md`。关键坑：

1. **动作维度写反**：环境要 `[acc, steer, 0]`，rollout 中通道颠倒 → 车持续画圆
2. **f_pred_batch 参考点 off-by-one**：动力学推进一步后参考点查询仍用旧索引 → heading error 180° 反向，400+ epoch 不收敛
3. **delta_phi 是 bearing error 不是 heading error**：`atan2(dy,dx)-ego` 应改为 `ref_h - ego_h`
4. **Wheelbase 硬编码 4.0m vs 真实车长 5.0m**：模型高估转弯能力 25%
5. **sqrt 梯度 NaN**：距离=0 时 `1/sqrt(0)=inf` → NaN 级联污染网络
6. **幽灵车**：GPUDrive 填零车被当真实车 → 虚假 penalty
7. **每步随机换候选路径**：参考点侧向跳动 7.5m → 追踪信号污染
8. **tracking_only 噪声不衰减**：σ=0.1 永不降 → 策略永远模糊

## 引用

- 原版 IDC 论文：[Integrated Decision and Control: Toward Interpretable and Computationally Efficient Driving Intelligence](https://arxiv.org/abs/2103.10290)
- GPUDrive：[GPUDrive: Data-driven, multi-agent driving simulation at 1 million FPS](https://arxiv.org/abs/2408.01584)
