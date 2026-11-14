# CHANGELOG

## 2026-11-14 — 单路径基线定版 + 工具化

### 训练结果（定版基线）

- `baseline-v1-2400ep`（commit `9b88ff0`）：75 world × 2400 epoch，14.2 小时，**一次未崩**
- PDMS 69.9（峰值 72.0）、完成度 83.7%、碰撞 42 步、压线 19 步、存活 73.8/75
- 跟踪：`frac<5m` 98~100%，`median pos_err` 0.3~0.5 m
- 240 次滚动替换全部成功，覆盖约 9000 个不同场景
- 关键发现：**约 300 epoch 即收敛**，后 2100 epoch 仅再涨 1.4 分
- 泛化：24 个新场景均值 61.4（40.6~75.4），约 42% 低于 60

### 修复

| 问题 | 根因 | 修法 |
|---|---|---|
| 换场景时容器 `Exited 134` | 换批候选池用了 `dense_files`（最稠密 1 万），`set_maps` 内部 kernel 启动资源不足 | 候选池改回全量 `all_files` 均匀抽 |
| 场景轮换不生效 | 只替换 bad/reached，没坏 world 就不换 | 改为确定性滚动替换（`resample_ratio`） |
| 部分 world 整轮不训练 | `good_worlds` 缓存跨 epoch 未失效 | 新增 `begin_epoch()` |
| PDMS 完成度失真 | EP 用步数时钟 | 改投影进度索引 `temp_idx_pos` |
| GIF 是静止图 | `time_steps` 传了回合号而非仿真步 | 改为离线渲染，彻底绕开 |

### 新增

- `src/utils/storyboard.py`：离线 storyboard 渲染（车道边界/参考虚线/ego 尾迹/周车尾迹/图例），
  单 world 约 8 秒，取代原来每 world 4 分钟以上的点线渲染
- eval 落盘 `traces/YYYYMMDD/trace_*.npz`（逐帧位姿 + 事件 + 地图 + 专家路径）
- `tools/overnight.sh`：崩溃后按 episode 编号取最新 checkpoint 自动续训
- `tools/plot_train_curves.py`、`tools/fetch_remote.py`、`tools/render_trace.py`、`tools/README.md`
- `tools/test_swap*.py` / `test_second_env.py`：记录 madrona 引擎与 `set_maps` 的边界行为

### 配置变更

- `viz.traj_plot_interval` 20 → **100**（2400 轮出图从 480 张降到 96 张）
- `world.resample_interval` 10、`world.resample_ratio` 0.5
- cost weight 回到论文值（`pos .04 / speed .01 / heading .1 / steer .1 / acc .005`）

### 已知限制（新增记录）

- 同一进程只能有一个 madrona 引擎（建第二个 env 会 abort）
- 世界数受显存限制：150 world 训练步 OOM，300 world 仿真步 OOM，**75 是安全值**
- 单轮耗时与世界数几乎无关，瓶颈是 `batch=128 × horizon=30` 的 BPTT
- 邻居为数据集回放（非反应式），结论须限定为"记录交通流中的单智能体规划"

---

## 2026-11-13 — 表示层重构与实现缺陷修复

### 修复（bug 级）

- 参考点由"步数时钟驱动"改为**最近点投影 + 单调索引**（`ref_window=25`）
- 道路 penalty 的 `ego_road_dist` 变负导致 `relu(D − 负数)` 爆到截断上限 → `clamp(min=0)`
- PDMS 的 EP 改为末步投影进度
- 各处截断常量统一（原本三处不一致导致梯度静默归零）
- 轴距改为逐样本读取 `size.length`（实测中位数 5.286 m，配置值 5.0 偏低 5.4%）
- 缓冲区的无穷/哨兵值防护（周车 phantom 槽位、损坏场景）

### 新增（表示层）

- `state_mode=ego`：车体系表示，逐帧平移+旋转
- `ref_mode=nearest` / `ref_window`：滑窗最近点投影
- `obs_norm`：`RunningNorm`（先验收缩 + clip），实测各维标准差相差 720 倍
- 状态新增 `veh_len`（轴距）维，总维度 65 → **66**
- `pev_interval=5`：不用的 Critic 降频，省下约 97% 的 rollout 算力

### 性能

- 单轮耗时 35 s → **19 s**（75 world）

---

## 2026-11-13 — 初版审查

- 按 Guan et al. 2023 IDC 逐条比对，确认 actor 的"模型推演 + 反向传播"即 GEP，
  不再是完整 OCP 求解
- 判定原实现的 RL 方法**属于**该论文方法，但实现有若干偏离（详见 `IDC_review.md`）
- 明确不做：候选路径 / critic 选路 / 安全护盾 / 红绿灯（超出修复范围）
