# IDC-Bridge

把 **IDC（Integrated Decision and Control）** 论文的算法跑在 **Waymo 开放数据集 + GPUDrive** 模拟器上：
单进程训练循环，75 个场景并行仿真，Actor 为可微模型推演（GEP），Critic 做策略评估。

> 当前定版为**单路径基线**：`git tag baseline-v1-2400ep`（commit `9b88ff0`）。
> 横向候选选择（多参考路径）尚未实现，设计讨论见文末「后续工作」。

---

## 0. 当前状态快照

| 项 | 值 |
|---|---|
| 表示 | `state_mode=ego`（车体系）、`ref_mode=nearest`（最近点投影，窗口 25）、`obs_norm=true` |
| 状态维度 | **66** 维（见 §4） |
| 训练规模 | 75 world × 2400 epoch，seed 5 |
| 场景轮换 | 每 10 epoch 滚动替换 50% 的 world，共 240 次，覆盖约 **9000 个不同场景** |
| 基线成绩 | PDMS **69.9**（峰值 72.0）、完成度 83.7%、碰撞 42 步、压线 19 步、存活 73.8/75 |
| 跟踪质量 | `frac<5m` 98~100%，`median pos_err` 0.3~0.5 m |
| 收敛速度 | 约 **300 epoch 即收敛**，之后 2100 epoch 仅再涨约 1.4 分 |
| 泛化（24 个新场景） | 均值 **61.4**，范围 40.6~75.4，约 42% 的场景低于 60 |

复现：

```bash
bash tools/overnight.sh      # 参数在脚本头部：TOTAL=2400 INTERVAL=10 RATIO=0.5 SAVE=20
```

---

## 1. 算法简介

IDC 的核心是**一边开一边学**：

- **PEV（策略评估）**：Critic 回归 H 步 cost-to-go，每 `pev_interval` 步更新一次（默认 5）。
- **PIM（策略改进）**：每 `pim_interval` 步（默认 30），Actor 用可微运动学模型推演 H=30 步，
  最小化 `J = J_actor + ρ·J_penalty`，梯度经动力学回传。ρ 用 EMA 跟踪违规度自动调节。
- **GEP**：不做完整 OCP 求解，而是把"模型推演 + 反向传播"当作策略改进算子。

Actor 为 **TransformerEncoder**（30 帧滑窗、4 头自注意力，3.0 s 历史上下文），Critic 为独立 MLP。
动作空间：加速度 `[-3, 1.5] m/s²`、方向盘转角 `[-0.6, 0.6] rad`。

---

## 2. 与论文的差异

| # | 差异 | 说明 |
|---|---|---|
| 1 | 网络结构 | Actor 用 TransformerEncoder（30 帧 × 4 头滑窗），Critic 独立 MLP；论文共享 latent |
| 2 | 训练架构 | 单线程主循环替代 30 个异步 learner |
| 3 | 环境 | Nocturne → GPUDrive（Waymo） |
| 4 | 参考路径 | 单个 expert 中心轨迹 + 曲率限速。**Waymo 无私车道级地图，偏移路径不是真实可行驶车道**，故未做多候选（见 §10.1） |
| 5 | 碰撞模型 | 前后双圆模型替代单圆 |
| 6 | 状态表示 | 车体系表示 + 滑窗最近点投影 + 运行时归一化（见 §4） |
| 7 | 性能优化 | 批量 tensor 操作 + GPU `ref_tensor` 预索引 + PEV 降频 |
| 8 | ρ 调节 | EMA（`ρ←0.95ρ+0.05×target`），取代论文的乘性放大 |
| 9 | 转向监督 | `bc_weight`：由 expert heading/pos 差分算近似转向，打破 RL 转向饱和 |
| 10 | 道路 penalty | `relu(D_road_safe − ego_road_dist)` 路径化，替代 edge-based |
| 11 | Progress 项 | `l -= progress_weight × v_lon×cos(δφ)×dt`，避免"龟速最优" |
| 12 | PDMS 评估 | CARLA 风格（NC×DAC×DDC × weighted avg），EP 用**投影进度索引**而非步数时钟 |
| 13 | Buffer | PER + 30 步窗口；跨场景轮换用**代际标签**拒采失效样本 |
| 14 | 配置 | YAML `_base` 继承 + CLI 覆盖 |

---

## 3. 快速开始

### 3.1 依赖

```bash
pip install gpudrive torch numpy matplotlib pyyaml imageio paramiko
```

必须在带 madrona/CUDA 的环境里跑（本仓库验证于 `gpudrive:latest` 镜像，
venv 在 `/gpudrive/.venv`）。

### 3.2 配置继承

```
src/configs/
├── base.yaml    # training/agent/dynamics/world/diag/metrics — 共享
├── train.yaml   # _base: base.yaml + env / paths / viz
└── eval.yaml    # _base: base.yaml + env(epochs=1) / paths / viz
```

优先级：**CLI > 子 YAML > base**。

### 3.3 训练

```bash
# 从零开始
python src/scripts/train/idc-train.py --data-dir /data/data_json/training \
    --file-dir /workspace/data --device cuda --epochs 400

# 从 checkpoint 续训（epochs 是"总轮数"，会自动减去已跑步数）
python src/scripts/train/idc-train.py --data-dir /data/data_json/training \
    --file-dir /workspace/data --device cuda --epochs 2400 \
    --load-model --model-path /workspace/data/checkpoints/20261113/xxx.pth

# 无人值守（崩了自动续训）
bash tools/overnight.sh
```

### 3.4 评估

```bash
python src/scripts/eval/idc-eval.py --data-dir /data/data_json/training \
    --model-path /workspace/data/checkpoints/20261113/xxx.pth \
    --file-dir /workspace/data --num-worlds 24 --epochs 1 --seed 0 \
    --gif-enabled true --gif-view-mode bird_2d --gif-max-worlds 3
```

### 3.5 离线密度扫描

```bash
python src/scripts/utils/scan_density.py --data-dir <tfrecords> \
    --output /workspace/data/density_full.json --batch-size 150
```

Waymo 70k 场景的密度分布（实测）：

| 密度 | 文件数 | 占比 |
|---|---|---|
| 0 | 381 | 0.5% |
| 1–2 | 3,275 | 4.7% |
| 3–5 | 7,780 | 11.1% |
| 6–10 | 13,984 | 19.9% |
| 11–20 | 19,565 | 27.9% |
| 21–30 | 10,155 | 14.5% |
| 31+ | 15,032 | 21.4% |

均值 19.19 / 中位数 15 / 范围 0–63。

---

## 4. 状态与表示

### 4.1 布局（66 维）

| 区间 | 维度 | 内容 |
|---|---|---|
| `[0:6]` | 6 | 自车：x, y, v_lon, v_lat(恒 0), phi, omega(恒 0) |
| `[6:38]` | 32 | 周车：8 车 × (x, y, phi, v) |
| `[38:46]` | 8 | 有效性 mask |
| `[46:64]` | 18 | ref_error：dp/dphi/dv + 3×(lat, dphi, road, spd) @ t+5/10/15 + curv/spd/road ahead（指数衰减聚合） |
| `[64:65]` | 1 | **投影进度索引** temporal |
| `[65:66]` | 1 | **轴距 veh_len**（取自 `size.length`） |

### 4.2 三个关键设计

1. **车体系表示（`state_mode=ego`）**：每帧把周车坐标平移+旋转到自车坐标系，
   消除绝对坐标漂移，也让策略与场景绝对位置解耦。
2. **最近点投影参考（`ref_mode=nearest`，`ref_window=25`）**：参考点按"离自车最近"在
   窗口内搜索，并对索引取单调 `max(prev, best)`，避免参考点随噪声抖动或倒退。
   > 这修掉了原先"参考点按步数时钟推进"导致的**参考点跑到自车前方几十米**的问题
   > （`idx_lag ≈ 0` 说明仍是时钟驱动，`idx_lag > 0` 说明已由位置驱动）。
3. **运行时归一化（`obs_norm`）**：`RunningNorm`（先验收缩、clip=10）逐维标准化。
   实测各维标准差相差 **720 倍**（`ego_y` 42.7 vs `curv_ahead` 0.059），不归一化无法训练。

编码后进入网络的维度为 **64**（`ENCODED_EGO=4` + 周车/参考特征压缩；`v_lat`、`omega` 恒为 0 已丢弃）。

---

## 5. 代价函数与权重

采用论文权重：

| 项 | 权重 |
|---|---|
| `pos_err_weight` | 0.04 |
| `speed_err_weight` | 0.01 |
| `heading_err_weight` | 0.1 |
| `steer_cost_weight` | 0.1 |
| `acc_cost_weight` | 0.005 |
| `progress_weight` | 0.05 |

截断（**所有用到该量的地方共用同一组常量**，否则梯度会静默归零）：

| 量 | 截断 |
|---|---|
| `pos_err` | 50 m |
| `lat_err` | 20 m |
| `utility` | 1e4 |
| `penalty` | 100 |

---

## 6. 场景管理

### 6.1 两套池子

- `all_files`：全量（默认 70172 个）打乱后的列表
- `dense_files`：按周车密度降序的前 `dense_sample_size`（默认 1 万）个

初始批由 `SceneDataLoader` 从**全量**均匀抽 `num_worlds` 个。

### 6.2 滚动替换（`resample`）

每 `resample_interval`（默认 10）个 epoch 触发一次：

1. 选出要替换的 world：坏掉的 / 已到终点的优先，不足 `resample_ratio`（默认 0.5）比例时
   用健康 world 补足（**确定性轮换**，不会因为"这几天没坏 world"就不换场景）
2. 从 **`all_files` 均匀抽**新场景，替换这批 world
3. `agent.bump_world_gen(replace_w)` 打代际标签，其余 world 的 buffer 经验继续有效

> **候选池必须用全量池，不能用 `dense_files`。**
> 实测：换批若切到"最稠密的那批场景"，`set_maps` 内部会因 kernel 启动资源不足
> 直接 abort（`CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`，容器 `Exited 134`，无 traceback）。
> 换到同一批或重复批则正常，说明 `set_maps` 本身可用，问题在"换到特定规模的场景批"。

### 6.3 不要每轮换场景

| 理由 | 实测 |
|---|---|
| 换一次要 30~40 s，而单轮只有 19 s | 每轮换 = 训练时间翻倍 |
| 全换会让整个 buffer 作废 | 代际标签下 100% 拒采 |
| 每个场景只被看 9.1 s（一个 Waymo 片段） | Actor 每 30 步才更新一次，学不到东西 |

推荐：**每 10 epoch 换 50%**。

---

## 7. 评估与可视化

### 7.1 PDMS

```
PDMS = (NC × DAC × DDC) × (EP×5 + TTC×5 + C×2 + LK×2) / 14

NC  : 碰撞 → 0
DAC : off-road > 3 步 → 0
DDC : |delta_phi| > 90° → 0（逆行）
EP  : 路线完成度（用投影进度索引进度，不是步数时钟）
TTC : 最小碰撞时间 / 4.0 s
C   : 舒适性（jerk）
LK  : 车道保持
```

训练日志：`[PDMS] score=69.9 completion=83.7% collisions=42 off_road=19 (surviving: 74/75)`
评估额外输出终端 ASCII 表、`pdms_plots/` 雷达图与柱状图、Rollout PDMS 预测分。

### 7.2 Storyboard 渲染（推荐）

eval 在 rollout 阶段**只收数据**（ego/周车世界位姿 + 事件 + 地图点云 + 专家路径），
结束后离线渲染成 GIF 与关键时刻拼图：

```bash
python tools/render_trace.py <file_dir>/traces/YYYYMMDD/trace_*.npz -o plots/story
```

画面内容：车道边界（深棕方块）、road_line / lane_center、**品红粗虚线专家参考**（画在 ego 尾迹之上）、
ego 带朝向矩形 + 蓝色尾迹 + 车头箭头、周车橙色矩形 + 尾迹；带完整图例，
标题显示 `t / 秒数 / OFF-ROAD / COLLISION`。
单个 world 约 8 秒出图。

### 7.3 旧的点线渲染

`--gif-view-mode bird_2d|bird_3d` 已改走 storyboard；
`agent_pov` 仍用 gpudrive 的 `plot_agent_observation`。

> 历史 bug（已修）：bird 视角曾把 `time_steps` 传成回合号 `epoch`，
> 而它要求的是"已记录轨迹里的第几步"，导致导出的 GIF 是静止图。

---

## 8. 工具集

见 [`tools/README.md`](tools/README.md)。常用：

| 脚本 | 用途 |
|---|---|
| `tools/selftest_idc.py` | 35 项断言的离线自检，**改完代码先跑** |
| `tools/overnight.sh` | 崩溃后自动续训的看护脚本 |
| `tools/check_config.py` / `stat_state.py` / `check_wheelbase.py` | 配置、状态量纲、轴距核对 |
| `tools/plot_train_curves.py` | 日志 → PDMS/损失/跟踪 四联图 + 分段均值 |
| `tools/fetch_remote.py` | 下载日志 / GIF / trace（凭据走环境变量） |
| `tools/render_trace.py` | trace npz → storyboard / GIF |

---

## 9. 关键参数速查

### 9.1 CLI

| 参数 | 说明 |
|---|---|
| `--data-dir` | Waymo 场景目录（必填） |
| `--config` | YAML 路径（默认按脚本位置推断） |
| `--model-path` / `--load-model` | checkpoint 路径 / 是否续训 |
| `--num-worlds` / `--epochs` / `--seed` | 并行世界数 / 总轮数 / 种子 |
| `--state-mode` / `--ref-mode` / `--obs-norm` | 表示层三个开关（评估时从 checkpoint 恢复） |
| `--resample-interval` / `--resample-ratio` | 场景滚动替换频率与比例 |
| `--max-bad-worlds` | 坏 world 数超过该值即触发替换（设大则只用 interval 触发） |
| `--min/max-partner-density` | 场景密度区间 |
| `--gif-*` | 可视化开关、视角、world 数、缩放、帧率 |
| `--no-road-penalty` / `--no-veh-penalty` | 诊断用，关闭对应约束 |

### 9.2 YAML 关键项

| 分组 | 关键参数 |
|---|---|
| `training` | `dt`, `horizon`, `batch_size`, `pim_interval`, `pev_interval`, `gamma`, `window_size`, `buffer_capacity`, `lr_*`, transformer 相关 |
| `agent` | 各 cost weight、`progress_weight`、`init_penalty`/`max_penalty`/`gep_amplifier`、`D_veh_safe`/`D_road_safe`、`bc_weight`、各种 `*_clamp` |
| `world` | `resample_interval`, `resample_ratio`, `max_bad_worlds`, `filter_threshold`, `density_cache_file`, `min/max_partner_density`, `dense_sample_size` |
| `viz` | `gif_enabled`, `gif_view_mode`, `gif_max_worlds`, `gif_zoom_radius`, `gif_fps`, `traj_plot_interval`, `traj_plot_max_worlds` |

---

## 10. 已知限制与坑

### 10.1 结构性问题

1. **一个进程只能有一个 madrona 引擎**：同进程建第二个 `GPUDriveTorchEnv` 会 abort
   （`setCudaHeapSize`）。因此"多卡数据并行"必须多进程，且每个进程独立持有 env。
2. **世界数受显存限制**：150 world 能起但训练步会 OOM（sim 与 BPTT 图共享 16 GB）；
   300 world 在 sim 里就 OOM。**75 是安全值。**
3. **单轮耗时与世界数几乎无关**（16/75/150 都约 19 s）——瓶颈是 `batch=128 × horizon=30`
   的 BPTT，不是仿真。
4. **邻居是数据集回放（非反应式）**：结论应限定为"记录交通流中的单智能体规划"。

### 10.2 踩过的坑（完整记录见 `info.md`）

| 坑 | 现象 | 修法 |
|---|---|---|
| 参考点按步数时钟推进 | 参考点跑到自车前方几十米，`pos_err` 被"前面"支配 | 改最近点投影 + 单调索引 |
| `good_worlds` 缓存跨 epoch 不失效 | `good=60` 但 `bad=0/reached=0`，15 个 world 整轮不训练 | 引入 `begin_epoch()` |
| 换批 abort | `Exited 134`，无 traceback | 候选池改全量（见 §6.2） |
| GIF 静止 | 90 帧完全相同 | `time_steps` 传步数而非回合号；现在改离线渲染 |
| PDMS 完成度失真 | 用步数时钟算 EP | 改投影进度索引 |
| `pos_err` 无界 | `ego_road_dist` 为负 → `relu(D − 负数)` 爆到截断上限 | `clamp(min=0)` |
| 转向 bang-bang 饱和 | 97% 打到边界 | expert 差分算近似 steer 做 BC |
| rho 累加器死锁 | 多轮迭代无解 | 改 EMA |
| 龟速最优 | 对称 `speed_err²` 惩罚慢速 | 降权重 + progress 项 |

### 10.3 不要做的事

- **不要用 `dense_files` 做换批候选池**（会 abort）
- **不要在同一进程里建第二个 env**
- **不要把 `pos_err` 的截断值改小**（会把梯度静默吃掉）

---

## 11. 后续工作：横向候选选择

现状：策略**只会减速、不会绕行**。证据是完成度从 88% 缓降到 82%、压线步数降到 9~19、
而碰撞稳定在 42~48 步——剩余 30 分全卡在"没有横向出口"。

要学到横向决策，三个条件必须同时成立：

1. 横向有可选性（≥2 条参考）—— 目前 `num_paths=1`
2. 代价按"被选中"的那条算 —— **管道已就绪**（`ref_tensor` 有 path 维、
   `path_indices` 全程透传、buffer 存 `_path_idx`），只是 `p_i` 恒为 0
3. Critic 变成 `V(s, τ)`，推理时 `argmin_τ` —— 小改动

建议先做 **L1 车道内避让**（±1.2 m 偏移，可行驶性天然成立），3 条候选，
探索用 ε-greedy（ε≈0.2），训练预算 400 epoch + `resample_interval 2`。

---

## 12. 引用

- IDC: [Integrated Decision and Control](https://arxiv.org/abs/2103.10290)
- GPUDrive: [Data-driven, multi-agent driving simulation at 1 million FPS](https://arxiv.org/abs/2408.01584)
