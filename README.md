# IDC-Bridge

把"集成决策与控制"（IDC, Integrated Decision and Control）论文的算法跑在 Waymo 开放数据集 + GPUDrive 模拟器上。

> **两个分支**：`main` = 改进版（Transformer + EMA rho + BC loss 等），`baseline` = 原论文方法（MLP + 经典 GEP，用于对比实验）。

## 算法简介（main 分支改进版）

IDC 的核心思想是**一边开一边学**：

- **PEV（策略评估）**：Critic 网络评估当前 state 的价值，每步更新。
- **PIM（策略改进）**：每隔 N 步，Actor 通过可微运动学模型推演未来 H 步轨迹，计算跟踪误差 + 碰撞风险的总成本，反向传播改进策略。ρ 通过 EMA 指数移动平均跟踪近期违规度，自动均衡。

Actor 为 **TransformerEncoder**（30 步滑窗 + 4 头自注意力，3.0s 历史上下文），Critic 保持独立 MLP。动作空间为加速度（[-3,1.5] m/s²）和方向盘转角（[-0.6,0.6] rad）。参考路径为 expert 中心轨迹 + 曲率限速，单路径训练。

## 和原论文的差异（以下描述 main 分支改进版）

1. **网络结构**：Actor 为 TransformerEncoder（30 帧 × 4 头自注意力滑窗，3.0s 历史），Critic 为独立 MLP（原论文共享 latent embed）
2. **训练架构**：单线程主循环替代 30 个异步 learner
3. **环境**：Nocturne → GPUDrive（Waymo 数据）
4. **参考路径**：expert 中心轨迹 + 曲率限速（单路径，无 lateral offset；Waymo 无私车道级地图，无法生成多条可行驶路径）
5. **碰撞模型**：前后双圆模型替代单圆
6. **状态空间**：65 维，含近前视点（t+5/10/15）+ 指数衰减远前视聚合特征（曲率/速度/道路宽度预期）
7. **性能优化**：批量 tensor 操作 + GPU `ref_tensor` 预索引
8. **EMA rho 调节**：ρ 通过指数移动平均（`ρ←0.95ρ+0.05×target`）跟踪违规度，自然均衡
9. **Steer BC loss**：从 expert heading/pos 差分计算近似转向监督（`atan(L×Δh/Δs)`），打破 RL 转向饱和
10. **YAML 层级配置**：`base.yaml`（共享）→ `train.yaml`/`eval.yaml` 继承覆盖
11. **道路 penalty 路径化**：`relu(D_road_safe - ego_road_dist)` 替代 edge-based
12. **权重重平衡**：`pos_err_weight=0.03`，`speed_err_weight=0.005`（降 10× 打破龟速最优解），`steer_cost_weight=0.15`（抑制高速抖动）
13. **Progress reward**：沿路前进正向奖励（`l -= 0.02 × v_lon×cos(δφ)×dt`），替代对称速度惩罚
14. **PDMS 评估**：CARLA 风格评分（NC×DAC×DDC × weighted avg）
15. **PER Buffer 窗口化**：每个经验存储完整 30 步窗口，~470 MB CPU 内存（单路径后）
16. **指数衰减前视聚合**：全部未来 90 步的加权平均曲率/速度/道路特征
17. **Eval 随机采样**：每次评估随机 seed，避免固定种子导致重复采样同一组世界

## 基线版本 (baseline 分支)

```bash
git checkout baseline
```

严格按原论文方法实现，用于和改进版对比实验：

| | baseline（原论文） | main（改进版） |
|------|:---:|:---:|
| Actor | MLP 3 层 512→512→256 | Transformer 30帧 |
| Critic | MLP 3 层（相同） | MLP 3 层 |
| GEP rho | 经典累加 `rho += amplifier_c` | EMA 指数移动平均 |
| 前视特征 | dp/dphi/dv/lat 仅 4 维 | 15 维（含衰减聚合） |
| State dim | 62 | 62 |
| Buffer | 单帧 window=1 | 30 帧窗口 |
| 辅助 loss | 无（BC/Speed/Progress 全无） | Steer BC + Speed BC + Progress |
| 配置 | `base_baseline.yaml` | `base.yaml` → `train.yaml` |
| 训练脚本 | `idc-train-baseline.py` | `idc-train.py` |

训练命令：
```bash
python src/scripts/train/idc-train-baseline.py --data-dir /path/to/data
```

参考路径：单条 expert 中心轨迹 + 曲率限速（与 main 分支相同，Waymo 无私车道级地图无法生成原论文的 Lane-Level 多条路径）。

## 快速开始

### 安装

```bash
pip install gpudrive torch numpy matplotlib pyyaml
```

### 配置文件

配置采用 **`_base` 继承机制**，共享参数放 `base.yaml`，train/eval 各自覆盖差异：

```
src/configs/
├── base.yaml      # training/agent/dynamics/world/diag/metrics — 共享
├── train.yaml     # _base: base.yaml + env(epochs=300, num_worlds=50)/paths/viz(gif=false)
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
# 统一训练（rho 使用 EMA 自动均衡，无需分阶段）
python src/scripts/train/idc-train.py --data-dir /path/to/waymo/data

# 从 checkpoint 继续训练
python src/scripts/train/idc-train.py \
    --data-dir /path/to/waymo/data \
    --model-path /path/to/checkpoint.pth

# 覆盖 YAML 参数
python src/scripts/train/idc-train.py \
    --data-dir /path/to/waymo/data \
    --num-worlds 50 --lr-actor 5e-5

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
| `--init-penalty` | 初始 ρ（0.01），EMA 锚点值 |
| `--num-worlds` | 并行仿真世界数 |
| `--epochs` | 训练轮数 |
| `--lr-actor` | Actor 学习率 |
| `--lr-critic` | Critic 学习率 |
| `--max-penalty` | ρ 上限（默认 3.0） |
| `--device` | 计算设备（cuda/cpu） |
| `--seed` | 随机种子 |
| `--density-cache-file` | 离线扫描的全量密度 JSON |
| `--min-partner-density` | 周车密度下限（含），0=不限 |
| `--max-partner-density` | 周车密度上限（含），默认 30 |
| `--dense-sample-size` | 候选池大小，0=不限 |
| `--no-road-penalty` | 关闭道路约束惩罚（诊断用） |
| `--no-veh-penalty` | 关闭周车碰撞惩罚（诊断用） |

### YAML 配置速查

完整参数见 `configs/base.yaml`，关键分组：

| 分组 | 包含参数 |
|------|----------|
| `env` | `num_worlds`, `max_agents`, `epochs`, `seed`, `device` |
| `training` | `dt`, `horizon`, `batch_size`, `hidden_dim`, `lr_actor_max`/`min`, `lr_critic_max`/`min`, `pim_interval`, `gamma`, `buffer_capacity`, `window_size`, `transformer_d_model`, `transformer_nhead`, `transformer_num_layers`, `transformer_dropout` |
| `agent` | `pos_err_weight`, `heading_err_weight`, 等 7 个 cost weight；`noise_std`/`decay_rate`/`min`；`init_penalty`/`max_penalty`/`amplifier_c`；`D_veh_safe`/`D_road_safe`/`half_length`/`half_width`；`bc_weight` |
| `dynamics` | `wheelbase`, `lr_ratio`, `v_max` |
| `world` | `min_partner_density`, `max_partner_density`, `dense_sample_size`, `max_bad_worlds`, `filter_threshold`, `dataset_size`, `density_cache_file` |
| `paths` | `file_dir`, `model_path`, `save_freq`, `load_model` |
| `diag` | `fix_speed`, `fix_heading`, `no_sign`, `no_road_penalty`, `no_veh_penalty` |
| `viz` | `gif_enabled`, `gif_fps`, `gif_record_interval`, `gif_max_worlds`, `gif_save_dir`, `gif_world_selection`, `gif_view_mode`, `gif_zoom_radius` |
| `metrics` | `pdms`: `nc_weight`, `dac_threshold`, `ttc_horizon`, `jerk_max`, `ep_weight`, `ttc_weight`, `comfort_weight`, `lk_weight` |

### 配置继承说明

| 文件 | 说明 |
|------|------|
| `base.yaml` | 6 个共享分组（`training`/`agent`/`dynamics`/`world`/`diag`/`metrics`） |
| `train.yaml` | `_base: base.yaml` + `env`(epochs=300)+ `paths` + `viz`(gif=false) |
| `eval.yaml` | `_base: base.yaml` + `env`(epochs=1, num_worlds=10) + `paths` + `viz`(gif=true) |

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
| `gif_enabled` | `true` | 是否生成 GIF（关闭可提速） |
| `gif_fps` | `15` | GIF **播放**速度（帧/秒） |
| `gif_record_interval` | `2` | **录制**间隔（每隔 N 个 env step 录一帧） |
| `gif_max_worlds` | `10` | 最多录制 world 数（0=全部） |
| `gif_save_dir` | `/workspace/data/gifs` | GIF 输出目录 |
| `gif_world_selection` | `first` | 录制 world 选择：`first`（前 N）/ `random`（随机 N） |
| `gif_view_mode` | `bird_2d` | 视图模式：`bird_2d` / `bird_3d` / `agent_pov` |
| `gif_zoom_radius` | `70` | 鸟瞰视图缩放半径 |

### PDMS 评分

训练每 epoch 和评估时会计算 PDMS（规划决策评分）：

```
PDMS = (NC × DAC × DDC) × (EP×5 + TTC×5 + C×2 + LK×2) / 14

NC:  碰撞 → 0
DAC: off-road > 3 步 → 0
DDC: |delta_phi| > 90° → 0（逆行）
EP:  路线完成度
TTC: 最小碰撞时间 / 4.0s
C:   舒适性（jerk < 10 m/s³）
LK:  车道保持（|lat| / road_width）
```

训练日志格式：`[PDMS] score=72.3 completion=89.2% collisions=3 off_road=12`

评估时额外输出：
- 终端 ASCII 评分表（各项得分、违规统计）
- `pdms_plots/pdms_radar_epoch*.png` — 加权项雷达图
- `pdms_plots/pdms_bar_epoch*.png` — 每个 world 的 Driving Score 柱状图
- Rollout PDMS：step 0 推演 Horizon 步的预测得分

## State 布局（65 维）

| 区间 | 维度 | 内容 |
|------|------|------|
| [0:6] | 6 | 自车: x, y, v_lon, v_lat, phi, omega |
| [6:38] | 32 | 周车: 8 车 × (x, y, phi, v) |
| [38:46] | 8 | validity mask |
| [46:64] | 18 | ref_error: dp/dphi/dv + 3×(lat+dphi+road+spd) @ t+5/10/15 + curv_ahead/spd_ahead/road_ahead |
| [64:65] | 1 | temporal index |

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
│   │   └── continuous_actor_critic.py  # TransformerActor + ContinuousCritic
│   ├── buffer/per_buffer.py
│   ├── scripts/
│   │   ├── train/idc-train.py
│   │   ├── eval/idc-eval.py
│   │   └── utils/scan_density.py   # 离线密度扫描
│   ├── metrics/
│   │   ├── pdms.py                  # PDMS 在线/rollout 评分
│   │   └── plotter.py               # 雷达图/柱状图/ASCII 表
│   └── utils/
│       ├── config.py               # YAML _base 继承 + CLI 合并
│       ├── traj_visualizer.py
│       └── ...
├── info.md
└── README.md
```

## 踩过的坑

完整记录见 `info.md`。关键坑：

1. **f_pred_batch 参考点 off-by-one**：动力学推进一步后参考点仍用旧索引 → heading error 180° 反向
2. **delta_phi 是 bearing error 不是 heading error**：`atan2(dy,dx)-ego` → `ref_h - ego_h`
3. **道路 penalty 公式反向**：旧公式惩罚路中心、奖励靠路边
4. **Steer bang-bang 饱和 97%**：BC loss 从 expert heading/pos 差分计算 approximate steer 打破饱和
5. **GEP rho 累加器死锁**：线性/乘性/双向/warmup 五轮迭代 → 最终改为 EMA
6. **3 条偏移路径降低训练质量**：Waymo 无私车道级地图，lateral offset 路径不是真实可行驶车道 → 改为单 expert center 路径
7. **Cost function 定义龟速为最优解**：`speed_err²×0.05` 惩罚慢速和超速同等力度 → 模型选择慢速保安全 → completion 卡 60% → 降 speed_err_weight 10× + 加 progress reward 打破
8. **BPTT 延长**：horizon/window 16→30（3.0s 前瞻），batch 256→128 控制显存。695 epoch 后 DDC 从 6/10→1/10，eval PDMS 从 20→51。证明 3s BPTT learnable 且改善方向判断
9. **高速转向振荡**：progress reward 推速度到 15m/s 后，原有 steer_cost=0.06 不足以抑制过冲 → 提高到 0.15

## 引用

- 原版 IDC 论文：[Integrated Decision and Control: Toward Interpretable and Computationally Efficient Driving Intelligence](https://arxiv.org/abs/2103.10290)
- GPUDrive：[GPUDrive: Data-driven, multi-agent driving simulation at 1 million FPS](https://arxiv.org/abs/2408.01584)
