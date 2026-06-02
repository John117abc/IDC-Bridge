# IDC-Bridge 开发记录：遇到的问题和解决方案

训练 100 个 epoch 从最初的 8 小时降到 ~1.1 小时，从频繁 NaN 崩溃到稳定收敛，经过以下问题修复。

---

## 1. 动作维度写反

**现象**：环境要 `[加速度, 转向角, 0]`，但 `select_action` 输出是 `[转向角, 加速度]`。模型推演和环境交互同时错。

**修复**：把 `torch.stack([delta_phy, a_phy])` 改成 `torch.stack([a_phy, delta_phy])`。

---

## 2. sqrt 梯度 NaN 导致训练崩溃

**现象**：增加 1-2 次 ρ 后 `compute_rollout_target` 输出的 target 变成 NaN，loss 爆炸。

**根因**：`torch.sqrt(x)` 在 x=0 时梯度 = `1/(2*sqrt(0))` = inf。当两车双圆心恰好重叠（距离=0），链式法则里分子也是 0，`inf × 0 = NaN`。这个 NaN 顺着 `clip_grad_norm_` 污染全部参数。

**修复**（3 处）：
- `kinematic_bicycle.py` 速度 sqrt：`sqrt(vx²+vy²)` → `sqrt(vx²+vy² + 1e-6)`
- `penalty_batch` 碰撞距离 sqrt：`+ 1e-8`
- `penalty_batch` 道路距离 sqrt：`+ 1e-8`
- `utility_batch` pos_err 和 speed_err sqrt：`+ 1e-8`
- 加 actor 权重 NaN 检测 + 自动回滚安全网

---

## 3. GPUDrive 幽灵车（phantom vehicles）

**现象**：penalty 恒定为 32.0，所有世界一模一样。世界 0 和 1 的车完全不动。

**根因**：Waymo 场景周车不足 8 辆时，GPUDrive 把空槽位填成 `[speed=0, rel_x=0, rel_y=0, rel_h=0]`。我们的状态构建把 `speed=0, rel_x=0, rel_y=0` 的幽灵车当成了真车（全局坐标 = ego 自身坐标）。8 辆幽灵车 × 4.0/车 = penalty=32.0 → clamp 到 10 → 梯度截断 → Actor 学不到任何方向。

**修复**：在 `get_idc_observations_batch` 里跳过 `speed==0 and rel_x==0 and rel_y==0` 的 partner。pad 补的零车在原点 (0,0)，距离 ego 数百米，不触发 penalty。

---

## 4. 双重 clamp 截断梯度（penalty + utility）

**现象**：去掉幽灵车后，多个世界仍然不动或飞出去。

**根因**：
- `clamp(l, -10, 10)` — 跟踪误差大（飞出去）= utility=-10 饱和 → Actor 看不到"往前能减小误差"
- `clamp(p, 0, 10)` — 8 车靠近 penalty=32 → clamp 到 10 → Actor 看不到"避开那辆车更安全"

**修复**：去掉组件级 clamp，只保留总计级 clamp：
- `compute_rollout_target`：去掉 `clamp(l, -10, 10)`，总计 `clamp(-250, 250)`
- `update_actor`：去掉 `clamp(l, -10, 10)` 和 `clamp(p, 0, 10)`，总计 `clamp(-100, 100)`

**后续调优**：去掉 per-step clamp 后发现飞出去的世界 utility=9.7M 炸全 batch → `compute_rollout_target` 回到 `clamp(l, -10, 10)`（防止极端世界污染 Critic），`update_actor` 也加回 `clamp(l, -10, 10)`。总计 clamp 同步缩小。

---

## 5. 碰撞约束过于敏感

**现象**：多车场景训练效果差，有些车在正常并排/跟车时也不动。

**根因**：
- `D_veh_safe=5.0`：两车并排邻车道（圆心距≈3.75m）也触发 penalty
- 32 对圆（8车×2×2）直接求和 → 多车必有 penalty → clamp 饱和 → 梯度截断

**修复**：
1. `D_veh_safe` 从 5.0 降到 **2.0**（只有真正逼近才触发）
2. `D_road_safe` 从 1.5 降到 **1.0**
3. 惩罚聚合从"32对求和"改成"每车取最危险的一对，再跨车求和"
   - `pair_pen.flatten(2).max(dim=2).values.sum(dim=1)`

改后的效果：
| 场景 | 圆心距 | 改前 penalty | 改后 penalty |
|------|--------|-------------|-------------|
| 并排邻车道 | 3.75m | 1.56 × 8 | 0（安全） |
| 跟车 3m 间距 | 3.0m | 4.0 × 8 | 0（安全） |
| 贴在一起 | 0m | 25→10（饱和） | 4.0（有梯度） |

---

## 6. 加速度映射默认负偏置

**现象**：路口低速起步的车（0.66 m/s）第一步就刹停，之后再没动过。

**根因**：原始映射 `a_phy = (norm+1)/2 * 4.5 - 3.0` 在 `norm=0` 时输出 **-0.75**。Actor 初始化随机偏置 ≈ -0.4 → a_phy = **-1.27** → 0.66 m/s 的车 0.5 秒刹停。

**修复**：分两阶段

**阶段 1 — 映射对称化（默认变 0）**：
```python
a_phy = torch.where(norm >= 0, norm * 1.5, norm * 3.0)
```
范围 [-3, 1.5] 不变，默认 0 → 滑行。

**阶段 2 — Actor 初始化加点正偏置**：
```python
nn.init.constant_(self.net[6].bias, 0.0)
self.net[6].bias.data[1] = 0.05  # tanh(0.05)≈0.05 → a≈0.075 m/s² 防刹停
```
默认轻微蠕行，不够强到撞墙—碰撞 penalty 梯度远超偏置。

**注意**：必须删旧 checkpoint 重训，新初始化才生效。

---

## 7. 贝塞尔路径恒定速度

**现象**：红绿灯路口专家停了，但参考速度恒定（全局均值），Agent 不知道要停。

**根因**：`_make_bezier_path` 里 `speeds = np.full(num_points, speed)` — 所有 91 步同一个速度。

**修复**：
- `generate_candidate_paths` 改为传专家的时变速度 `expert_vel.norm(dim=-1).cpu().numpy()`
- `_make_bezier_path` 改为接收数组，直接使用：`speeds = np.asarray(expert_speeds)`

---

## 8. 性能优化

**现象**：每 epoch ~5 分钟，100 轮 ~8.3 小时。

**瓶颈根源**：`f_pred_batch` 里的 per-sample Python 循环 — 每个样本独立调 `get_ref_state`（numpy→Python float），然后 `torch.tensor` 回 GPU。256×25=6400 次 Python→C 往返。

**修复**：

### 优化 1 — 批量参考查找（核心加速 ~10-20×）

改前：
```python
for i in range(256):
    ref_numpy = self.state_builder.get_ref_state(...)  # 逐个 numpy
    ref_tensor = torch.tensor(ref_numpy, ...)          # 逐个回 GPU
```

改后：
```python
refs = self.state_builder.get_ref_states_batch(w_i, x_next, y_next, ...)  # 一次批量
dx = refs[:, 0] - x_next  # 全程向量化
```

### 优化 2 — 道路张量缓存

`get_road_edges_batch` 每步推演从 GPU 拉一次全道路 map（25次/rollout）。加 `_road_cache`，每集只拉一次。

### 效果

| | 改前 | 改后 |
|------|------|------|
| Critic 推演单次 | ~3.0s | ~0.3s |
| 每 epoch | ~300s | ~40s |
| 100 epoch | ~8.3h | **~1.1h** |

---

## 9. actor_loss 迭代爆炸（target=1000 反复撞天花板）

**现象**：去掉 per-step clamp 后，某些世界飞出去几公里，单步 utility=400K → target 撞 clamp(-1000,1000) → Critic 分不清"偏离 20m"和"偏离 200m"。

**修复过程**：来回调了几轮 clamp 值，最终稳定在：
- `compute_rollout_target` 单步：`clamp(l, -10, 10)` ← 防止极端世界污染
- `compute_rollout_target` 总计：`clamp(total, -250, 250)` ← 25×10=250
- `update_actor` 单步：`clamp(l, -10, 10)` ← 防止极端世界炸 Actor 梯度
- `update_actor` 总计：`clamp(total_cost, -100, 100)`

Penalty 不 clamp（保持碰撞梯度），只靠总计 clamp 兜底。

---

## 10. Critic value 爆炸到 4300+

**现象**：训练中 Critic value max 从 115 跳到 4296，loss 从 4000 跳到 19.8 万，之后持续震荡无法恢复。

**根因**：Critic 只有裸 `Linear+ELU`，无 LayerNorm。异常坐标（15575m）进入时，随机权重 × 大输入 = 爆炸输出。巨大 MSE 梯度破坏 Critic 权重，级联污染后续所有世界。

```python
# 改前（裸奔）
class ContinuousCritic(nn.Module):
    def __init__(self, state_dim, hidden_dim=256, output_dim=1):
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, output_dim)
```

**修复**：Critic 改为 `Sequential(LN+ELU)×2 + Linear`，与 Actor 架构对齐。

```python
# 改后（受保护）
class ContinuousCritic(nn.Module):
    def __init__(self, state_dim, hidden_dim=256, output_dim=1):
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim),
        )
```

异常输入被 LayerNorm 归一化后，value max 从 4300 降到 ~24。

---

## 11. 15575m 误差来源诊断

**现象**：`[DIAG-ref] max pos_err=15575m` 反复出现在约 35 个世界。`[LARGE-ERR]` 显示 `ego=(-11000,-11000)`。

**诊断过程**：加了 5 种分层诊断标签逐层定位：
- `[PATH-RANGE]`：训练开始时打印路径起点/终点坐标，确认坐标系
- `[LARGE-ERR]`：`get_idc_observations_batch` 中 pos_err > 100m 时打印 ego/ref
- `[FPRED-CLAMP]`：推演中位移异常时打印
- `[FPRED-ERR]`：推演中 delta_p > 100m 时打印
- `[TELEPORT]`：环境步后 abs(x) > 1e5 时打印

**结论**：GPUDrive 使用本地坐标系（非 UTM），所有正常坐标在 ±100 范围内。`ego=(-11000, -11000)` 出现在约 35 个世界，是特定数据在 episode 末尾的默认标记值，不是推演漂移产生。

---

## 12. f_pred_batch clamp 策略演进

| 版本 | 锚点 | 窗口 | 行为 | 缺陷 |
|------|------|------|------|------|
| v1 | 相对上一步 `prev_x` | ±10m | 每步允许 10m 位移 | 25 步累积可达 250m |
| v2 | 相对参考路径 `ref_x` | ±50m | 异常从第一步就被截断 | 窗口外 grad=0，切断梯度链路 |
| v3 | 相对参考路径 `ref_x` | **±150m** | 保留梯度链路，仍拦截 15575m | — |

关键设计：
- ref 查表必须在 clamp **之前**（需要 ref 作为锚点坐标）
- clamp 后用 clamped 坐标计算 ref error
- 窗口外 grad=0，不污染 Actor 梯度
- 150m 窗口：20 步正常推演最大位移 < 30m，预留 5 倍裕量

```python
# v3 核心逻辑
refs = get_ref_states_batch(w_i, x_raw.detach(), y_raw.detach(), ...)
ref_x, ref_y = refs[:, 0], refs[:, 1]
x_next = torch.clamp(x_raw, ref_x - 150, ref_x + 150)
y_next = torch.clamp(y_raw, ref_y - 150, ref_y + 150)
```

---

## 13. 无效世界过滤（已回滚）

**误判**：尝试用 `abs(path_start) < 5000` 过滤无效世界，但 GPUDrive 所有世界都是本地坐标（±100 内），导致 200/200 世界全被过滤。

**回滚**：删除 bad_worlds 检测和过滤逻辑，恢复全部世界参与训练。仅保留 `[LARGE-ERR]` 诊断。

**教训**：不了解坐标系时不要做硬过滤。GPUDrive 使用本地（relative）坐标，不是 UTM 绝对坐标。

---

## 14. 道路 penalty 断崖 + Actor 画圆（核心发现）

**现象**：Actor loss 从 58 缓慢上升到 70，车画标准圆，不跟踪路径。Critic 收敛良好但 Actor 不收敛。

**根因链**：

```
车在推演中偏航离开道路
  ↓
旧道路 penalty = relu(1.0 - edge_dist)²
  edge_dist > 1m → relu(负数) = 0 → 梯度归零
  ↓
utility clamp(l, -10, 10)：pos_err > 5m 即饱和 → 梯度归零
  ↓
唯一幸存梯度 = control cost
  ↓
Actor 学到"不动就好" → 恒常 steer → 画圆
```

**修复**：

| 改动 | 当前 | 改为 | 原因 |
|------|------|------|------|
| 道路 penalty | `relu(1.0 - edge_dist)²` | **`relu(edge_dist - 5.0)²`** | 偏离越远惩罚越大，始终有梯度拉回 |
| `D_road_safe` | `1.0` | **`5.0`** | 5m 道路宽容区，正常行驶不触发 |
| utility 单步 clamp | `(-10, 10)` | **`(-50, 50)`** | 允许 20m~50m 偏航有区分度 |
| Actor total clamp | `(-100, 100)` | **`(-500, 500)`** | 配合 20 步 × 50 上限 |
| Critic total clamp | `(-250, 250)` | **`(-500, 500)`** | 同步放宽 |

新道路公式语义：

```
edge_dist ≤ 5m（路上）: relu(-)² = 0             ← 零惩罚
edge_dist = 10m（偏航）: relu(5)² = 25            ← 有惩罚 + 有梯度
edge_dist = 100m（飞远）: relu(95)² = 9025        ← 强拉力回道路
```

对比旧公式 `relu(1.0 - edge_dist)²`：
| 场景 | 旧 | 新 |
|------|----|----|
| 正常在路 | 0 | 0 |
| 偏航 10m | **0（梯度死亡）** | 25（梯度存活） |
| 偏航 100m | **0（梯度死亡）** | 9025（强梯度） |

---

## 15. Actor 转向锁定 + ρ 污染

**现象 1**：20 个世界的 `norm_steer` 全为负（右转），训练后期不改。

**根因**：探索噪声 `std=0.05` 太小（转向噪声仅 `0.02 rad`），Actor 永远采样不到左转样本 → 策略锁定。

**修复**：
- 噪声 std：`0.05` 硬编码 → **`self.noise_std=0.2`**，PIM 后衰减 `×0.97`，下限 `0.05`
- Actor 学习率：`1e-5` → **`8e-5`**

```python
# select_action
noise = torch.normal(0, self.noise_std, ...)

# update_actor 末尾
self.noise_std = max(self.noise_std_min, self.noise_std * self.noise_decay_rate)
```

**现象 2**：ρ 快速增长到 100，penalty 淹没 utility 信号 → Actor 追移动靶。

**修复**：
| 参数 | 旧 | 新 | 原因 |
|------|---|----|------|
| `max_penalty` | `100` | **`10`** | 防止 penalty 淹没 utility |
| `heading_err_weight` | `0.3` | **`0.6`** | 加强转向纠正 |
| `rollout horizon` | `25` | **`20`** | 缩短梯度链 |

---

## 16. 其他参数调整

| 参数 | 旧 | 新 | 原因 |
|------|---|----|------|
| `pos_err_weight` | `0.04` | `0.2` | 增强跟踪信号 |
| `heading_err_weight` | `0.1` | `0.6` | 加强转向纠正 |
| padding 坐标 | `(0, 0)` | **`(1e6, 1e6)`** | 消除原点 ghost 车的虚假 penalty |
| `speeds` 源 | 恒定均值 | 专家时变速度 | 红绿灯能减速停车 |

---

## 17. Actor 输出通道在 rollout 中颠倒（致命 bug）

**现象**：修复所有已知问题后，tracking-only 模式下直线车道仍无法跟踪，车持续画圆，pos_err 15500m+。

**根因**：两个独立问题叠加：

### 17a. 通道语义颠倒

Actor 输出为 `[steer_raw, acc_raw]`（通道 0=转向，通道 1=加速度）。但 `KinematicBicycleModel.forward` 内部按 `action[0]=acc, action[1]=steer` 读取。rollout 直接喂 raw Actor 输出，不做任何通道重排 → 加速通道被当转向，转向通道被当加速。

加速 bias=0.05 → 动力学把它当 steering=0.05 rad → **车持续画圆**。

### 17b. raw tanh 未映射到物理量

`select_action` 把 `[-1,1]` 映射到 `acc [-3.0,1.5] m/s², steer [-0.4,0.4] rad`，但 f_pred_batch 直接喂 raw tanh，rollout 动力学用的物理尺度完全错误。

### 修复

在 `f_pred_batch` 动力学调用前加 4 行，与 `select_action` 保持一致：

```python
delta_phy = actions[..., 0] * 0.4
a_phy = torch.where(actions[..., 1] >= 0,
                    actions[..., 1] * 1.5,
                    actions[..., 1] * 3.0)
actions_phy = torch.stack([a_phy, delta_phy], dim=-1)
ego_next = self.dynamics(ego, actions_phy)
```

**必须删旧 checkpoint 重训。**

---

## 18. utility_batch 算在动作之前而非之后（致命 bug）

**现象**：通道 bug 修好后，tracking-only 仍不收敛，pos_err 100-130m 缓慢上升。

**根因**：`update_actor` 和 `compute_rollout_target` 循环中 utility 在 f_pred **之前**计算：

```python
# 原顺序（错误）
for t in range(horizon):
    u = actor(s)
    l = utility_batch(s, u, ...)   # s 是动作前的状态 — pos_err 等不依赖 u！
    s = f_pred_batch(s, u, ...)
```

`pos_err`, `heading_err`, `speed_err` 全从 s 读取，与 u 无关。仅 `steer_cost` + `acc_cost`（权重 0.1, 0.005）依赖 u。Actor 唯一梯度信号 = "输出 0 以最小化控制代价"。

### 修复

翻转顺序，utility 计算在 f_pred **之后**：

```python
# 修正后
for t in range(horizon):
    u = actor(s)
    s = f_pred_batch(s, u, ...)    # 先推演
    l = utility_batch(s, u, ...)    # 在新状态上算跟踪误差
```

梯度链路：`Actor 参数 → u → dynamics → s_next → tracking_err → loss`。

---

## 19. 梯度截断链（3 个 clamp 同步修复）

**现象**：修复 issue 17+18 后 pos_err 仍在 100-130m 停滞。

**全量审计发现**：

| # | 位置 | 原值 | 触发条件 | 影响 |
|---|------|------|---------|------|
| 19a | `update_actor` utility clamp | `[-50, 50]` | pos_err > 15.8m | 正常跟踪梯度截断 |
| 19b | `f_pred_batch` 位置 clamp | `ref ± 40m` | rollout 偏离 ref >40m | 位置梯度截断 |
| 19c | 动力学速度 clamp | `±30 m/s` | 极少触发 | 安全网保留 |

### 修复

- 19a：`clamp(l, -50, 50)` → `clamp(l, -5000, 5000)`
- 19b：直接删除 — 坏 world 已由训练脚本外部过滤
- 19c：保留 — `clip_grad_norm(max_norm=1.0)` 做终极梯度保护

---

## 20. GPUDrive 数据世界过滤体系

**现象**：30-35% 世界存在坐标异常（ego=(−11000,−11000) 或参考路径坐标系错位），产生 15560m+ 误差。

**根因**：GPUDrive 本地坐标框架 bug，非车辆物理运动产生。

### 两级过滤

**阶段 1 — 预训练路径坐标检测（一次）**：
```python
if np.max(np.abs(path_pos[:,0])) > 5000 or np.max(np.abs(path_pos[:,1])) > 5000:
    bad_worlds.add(w)
```

**阶段 2 — 每步 ego 坐标检查**（在 `--no-sign`、DIAG-ref、buffer 插入之前）：
```python
if abs(states[w][0]) > 5000 or abs(states[w][1]) > 5000:
    bad_worlds.add(w)
```

**数据流顺序**：`get_idc_observations → per-step filter → --no-sign → DIAG-ref → buffer`。异常数据永不流入 downstream。

**诊断验证**：同步打印 `delta_p`，确认所有被过滤世界的 delta_p 均 >15000m，无误杀。

---

## 21. heading 过冲震荡（"画龙"）

**现象**：跟踪开始工作但车大幅左右摆动跨多个车道，pos_err 随 step 累积增长（epoch 初 3.75m → 末 50m）。

**根因**：`heading_err_weight=0.6` vs `steer_cost_weight=0.1`，梯度比 18:1。Actor 学到猛烈转向修正，动力学惯性延迟必然导致过冲震荡。

### 修复

| 参数 | 旧 | 新 | 原因 |
|------|---|----|------|
| `heading_err_weight` | 0.6 | **0.3** | heading 修正更温和 |
| `steer_cost_weight` | 0.1 | **0.3** | 大转向代价更高 |

梯度比从 18:1 → ≈2:1。

---

## 22. delta_phi 计算修正

**现象**：车辆距参考点较远时，路径切线方向的 heading 参考缺乏意义。

**修复**：`delta_phi = atan2(dy, dx) - ego_θ`（参考点方向 vs ego），而非 `ego_θ - ref_θ`（路径切线方向 vs ego）。在 `_calc_ref_error`、`f_pred_batch`、`utility_batch` 三处同步。

---

## 23. 道路 penalty 线性化

**现象**：偏离道路后旧 penalty `relu(1.0 - edge_dist)²` 在 `edge_dist > 1m` 时输出 0 → 梯度死亡。

**修复**：`road_violation = F.relu(edge_dist - D_road_safe)`，线性始终有梯度。`D_road_safe=2.0`。

---

## 24. DIM_VALIDITY + DIM_TEMPORAL

- `DIM_VALIDITY`（8 维）：padding 车辆的 0/1 mask，防 LayerNorm 混淆真实特征
- `DIM_TEMPORAL`（1 维）：step-counter 时序索引，替代空间最近点搜索（消除弯道跳变）

状态维度：6+32+8+3+1 = **50 维**。

---

## 25. 噪声与学习率最终值

| 参数 | 值 | 说明 |
|------|-----|------|
| `noise_std` | 0.1（tracking-only）/ 0.15（正常） | 探索噪声，PIM 后 x0.98 衰减至 min 0.03 |
| `lr_actor` | 8e-5 | Adam |
| `lr_critic` | 3e-4 | Adam |
| `horizon` | 20 | rollout 步数 |
| `dt` | 0.4 | 动力学步长（秒） |

---

## 当前关键参数速查

| 参数 | 值 | 说明 |
|------|-----|------|
| state dim | **50** | ego(6) + others(32) + validity(8) + ref_error(3) + temporal(1) |
| `pos_err_weight` | 0.2 | 位置误差权重 |
| `speed_err_weight` | 0.01 | 速度误差权重 |
| `heading_err_weight` | **0.3** | 朝向误差权重 |
| `steer_cost_weight` | **0.3** | 转向控制代价 |
| `acc_cost_weight` | 0.005 | 加速控制代价 |
| `D_veh_safe` | 2.0 | 双圆心碰撞阈值 |
| `D_road_safe` | 2.0 | 道路安全距离 |
| `HALF_L` | 2.25 | 半车长 |
| `HALF_W` | 1.0 | 半车宽 |
| `noise_std` | 0.1（tracking）/ 0.15（正常）→ 0.03（×0.98/PIM） | 探索噪声 |
| `lr_actor` | 8e-5 | Actor 学习率 |
| `lr_critic` | 3e-4 | Critic 学习率 |
| `dt` | 0.4 | 时间步长（秒） |
| `horizon` | 20 | 推演步数 |
| `batch_size` | 256 | PEV/PIM 采样数 |
| `num_worlds` | 200 | 环境世界数 |
| `max_penalty` (ρ 上限) | 10 | 最大惩罚系数 |
| `amplifier_c` | 1.015 | ρ 每次 PIM 放大倍率 |
| `pim_interval` | 30 | PEV 步数到 PIM |
| `clamp` utility 单步 | **-5000/+5000** | 不截断正常跟踪梯度 |
| `clamp` f_pred 位置 | **已删除** | 坏 world 由训练脚本过滤 |
| 动力学速度 clamp | ±30 m/s | 安全网 |
| Actor acc bias | 0.05（tanh 后约 0.075 m/s²） | 防刹停 |
| f_pred 通道映射 | raw_steer×0.4→rad, raw_acc×1.5/3.0→m/s², stack([acc,steer]) | 与 select_action 一致 |
| delta_phi | `atan2(dy, dx) - ego_θ` | 参考点方向 |
| 道路 penalty | `relu(edge_dist - 2.0)` 线性 | 始终有梯度 |
| 坏世界过滤 | 阶段1 预训练路径检查 + 每步 ego 检查 | 异常值不进 buffer |

---

## 日志诊断体系

| 诊断标签 | 位置 | 级别 | 触发条件 |
|----------|------|------|----------|
| `[FILTER-path]` | 预训练过滤 | WARNING | 路径坐标 >5000 |
| `[FILTER-ego]` | 每步过滤 | WARNING | ego 坐标 >5000，同步打印 delta_p |
| `[DIAG-ref]` | 每 5 步 | INFO | 最大 pos_err（仅正常 world） |
| `[DIAG-init]` | epoch step 0 | INFO | 初始速度、动作、参考速度 |
| `[DIAG-act]` | select_action | INFO | 每 10 步动作值 |
| `[DIAG-critic]` | compute_rollout_target | INFO | 每 50 步 utility 范围 |
| `[DIAG-pen]` | update_actor | INFO | 每次 PIM step 0 penalty 范围 |
| `[TRACK-DIAG]` | utility_batch | DEBUG | pos_err > 5m 时最差样本 |
| `[FPRED-ERR]` | f_pred_batch | WARNING | delta_p > 100m |
| `[LARGE-ERR]` | get_idc_observations_batch | DEBUG | pos_err > 100m |
| `[TELEPORT]` | idc-train.py | WARNING | abs(x\|y) > 1e5 |
| `[NaN]` | 各方法 | WARNING | tensor 含 NaN |
| `回合 X/Y, 步数 Z/91` | idc-train.py | INFO(10步)/DEBUG | 每步 |

---

## 26. f_pred_batch 参考点时间索引 off-by-one（致命 bug）

**现象**：修复 issue 17-25 后 tracking-only 训练 400+ epoch，直道跟踪仍很差，弯道完全无法跟随。

**根因**：`f_pred_batch:228` 动力学把自车推进一步（step N→N+1），但参考点查询仍用旧索引 `temporal_idx`（step N）：

```python
temporal_idx = states[:, ...]         # step N
temporal_next = temporal_idx + 1      # step N+1
ego_next = self.dynamics(ego, action) # 自车从 N → N+1

# BUG: 用了旧索引
refs = get_ref_states_batch(..., temporal_indices=temporal_idx)  # ← 应该是 temporal_next
```

**直道上的后果**（v=10 m/s, dt=0.1s, 每步 1m）：

```
dx = ref_x[step N] - ego_x[step N+1] = -1.0m  ← 参考点在车后
delta_phi = atan2(0, -1) - theta ≈ ±π rad    ← 180° 方向反了
heading_err² ≈ π² ≈ 9.87
utility contribution = 0.3 × 9.87 ≈ 2.96      ← 每步的巨大虚假惩罚
```

对比正常追踪 pos_err=1m 时仅有 `0.3 × 1 = 0.3`，heading 惩罚是 pos 的 10 倍且方向反了。

**修复**：`temporal_indices=temporal_idx` → `temporal_indices=temporal_next`（一行改动）。

**影响**：这是训练 400+ epoch 无收敛的**首要原因**。heading error 信号与实际跟踪目标相反，梯度冲突导致 Actor 无法收敛。

---

## 27. delta_phi 从 bearing error 改为 true heading error

**现象**：弯道上即使完美追踪也有非零 "heading error"。

**根因**：原 `delta_phi = atan2(dy, dx) - ego_theta` 是 bearing error（自车指向参考点的方向 − 自车方向），不是 heading error。弯道上参考点的方向和当前位置路径切向不同 → 虚假惩罚。

**修复**（2 处）：

| 文件 | 位置 | 改动 |
|------|------|------|
| `idc_state_builder.py` | 454 | `atan2(dy,dx) - ego[4]` → `ref_state[4] - ego_state[4]` |
| `idc_agent.py` | 264 | `atan2(dy,dx) - theta_next` → `refs[:,4] - theta_next` |

改后 `delta_phi = ref_heading - ego_theta`（真正航向对齐误差）。

---

## 28. Wheelbase 硬编码 4.0m vs GPUDrive 真实车长

**现象**：直道改善后，弯道仍转向不足。

**根因**：GPUDrive `forwardKinematics` 转弯公式 `w = v*cos(β)*tan(δ) / size.length`，其中 `size.length` 取自 Waymo 真实车辆长度（~4.3-5.2m）。IDC `KinematicBicycleModel` 硬编码 `L=4.0m`。

| 参数 | IDC 模型 | GPUDrive 实际 | 偏差 |
|------|---------|-------------|------|
| 有效 Wheelbase | 4.0 m | 4.3-5.2 m | +7-30% |
| 转弯速率（同 δ） | v·tan(δ)/4.0 | v·tan(δ)/5.0 | **模型高估 25%** |

Agent 在训练中认为自己能转过某弯，但真实环境转不过去。

**修复**：`kinematic_bicycle.py:35` `L: float = 4.0` → `5.0`。

---

## 29. Steering 最大值 0.4 → 0.6 rad

**根因**：GPUDrive C++ 对 steering **没有硬上限**（`forwardKinematics` 不做 clamp），±0.4 rad 完全是 IDC 自己加的。以 L=5.0m 计算：R_min = 5.0/tan(0.4) ≈ 11.8m，覆盖不了城市交叉口急弯（常见 5-8m）。

**修复**（3 处）：

| 文件 | 行 | 改动 |
|------|-----|------|
| `idc_agent.py` | 124, 217 | `* 0.4` → `* 0.6` |
| `action_mapper.py` | 14 | `steer_range=(-0.6, 0.6)` |

转弯半径从 ~9.5m（L=4.0, δ=0.4）→ ~6.8m（L=5.0, δ=0.6）。

---

## 30. 候选路径索引：步级随机 → episode 级固定

**现象**：3 条候选 Bezier 路径（3.75m 侧向偏移），每步随机选一条 → 相邻步参考点侧向跳动 7.5m → 追踪信号严重污染。

**修复**：`idc-train.py:152-163`，每个 episode 开始时为每个 world 随机选一个 pid ∈ {0,1,2}，整个 episode 内不变。跨 episode 重新随机，保留数据多样性。

---

## 31. 动态道路宽度（从 GPUDrive 读取替代硬编码 3.75m）

**根因**：硬编码 `lane_width=3.75` 不适用于窄路/宽路，offset 路径可能跑出道路边界。

**修复**：`idc_state_builder.py:50-77`，从 `map_observation_tensor[:,:,3]`（`segment_width`）读取每个 world 的实际车道宽度。invalid（≤0/NaN）时 fallback 到 3.75。

---

## 32. 前瞻参考点 — State 维度 50 → 54

**根因**：原 Agent 对前方弯道"蒙眼"——只知道当前误差（`delta_p`, `delta_phi`, `delta_v`），不知道 10 步后有 90° 弯道。弯道跟踪差的关键原因是缺乏路径预览信息。

**新增 ref_error 布局 (7 dims)**：

| 索引 | 含义 | 计算方式 |
|------|------|---------|
| [0] | `delta_p(t)` | 当前横向位置误差（带符号） |
| [1] | `delta_phi(t)` | 当前航向误差 |
| [2] | `delta_v(t)` | 当前速度误差 |
| [3] | `lat_l1` | t+3 横向误差（纯 lateral = `dy·cos(rh) - dx·sin(rh)`） |
| [4] | `dphi_l1` | t+3 航向误差 |
| [5] | `lat_l2` | t+6 横向误差 |
| [6] | `dphi_l2` | t+6 航向误差 |

**关键设计**：
- 前瞻横向误差用纯 cross product（`dy·cos(rh) - dx·sin(rh)`），避免直道上因前瞻距离大而误报大误差
- 前瞻索引 clamp 到 `num_pts-1`，终点处多个前瞻汇聚到同一点，自然退化
- 前瞻权重 `lookahead_pos_weight=0.05, lookahead_heading_weight=0.05`，比当前误差（0.3）小 6 倍，作为辅助梯度
- 不加速度前瞻（速度 profile 已按时序给）

**涉及文件**：
| 文件 | 行 | 改动 |
|------|-----|------|
| `idc_agent.py` | 36 | `DIM_REF_ERROR = 3 → 7` |
| `idc_agent.py` | 61-62 | 新增前瞻权重 |
| `idc_agent.py` | 270-286 | `f_pred_batch` 查询 t+3、t+6 参考点 |
| `idc_agent.py` | 412-413 | `utility_batch` 加入前瞻成本 |
| `idc_state_builder.py` | 249-264 | 初始 state 构建加入前瞻误差 |

**State 布局（54 维）**：

| 区间 | 维度 | 内容 |
|------|------|------|
| [0:6] | 6 | ego: x, y, v_lon, v_lat, phi, omega |
| [6:38] | 32 | others: 8 cars × 4 (x, y, phi, v) |
| [38:46] | 8 | validity mask |
| [46:53] | 7 | ref_error（见上表） |
| [53:54] | 1 | temporal index |

---

## 33. 噪声衰减扩展到 tracking_only 模式

**现象**：tracking_only 模式下 `noise_std=0.1` 永不衰减 → 环境执行始终带噪声 → 策略永远不干净 → 直线蛇形（画龙）。

**修复**：`idc_agent.py:344` 去掉 `if not self.tracking_only:` 守卫，所有模式统一衰减 `σ × 0.98 / PIM`，最低 0.03。

---

## 当前关键参数速查

| 参数 | 值 | 说明 |
|------|-----|------|
| state dim | **59** | ego(6) + others(32) + validity(8) + ref_error(12) + temporal(1) |
| ref_error | `[dp, dphi, dv, lat+road×3 at t+3/6/9]` | 当前 + 前瞻 3 点（含道路边界预览） |
| `noise_std` | 0.1 → 0.03（×0.98/PIM） | 探索噪声（统一衰减，不区分模式） |
| `lr_actor` | 8e-5 | Actor 学习率 |
| `lr_critic` | 3e-4 | Critic 学习率 |
| `dt` | 0.1 | 时间步长（秒） |
| `horizon` | 20 | 推演步数 |
| `batch_size` | 512 | PEV/PIM 采样数 |
| `num_worlds` | 200（A4000 16GB）/ 300（32GB+） | 并行训练世界数 |
| `dataset_size` | **0 = 自动检测 data_dir 文件数** | 重采样候选池大小 |
| `max_bad_worlds` | 100 | 坏世界数触发重采样阈值 |
| `max_penalty` (ρ 上限) | 10 | 最大惩罚系数 |
| `amplifier_c` | 1.015 | ρ 每次 PIM 放大倍率 |
| `pim_interval` | 30 | PEV 步数到 PIM |
| `clamp` utility 单步 | **-5000/+5000** | 不截断正常跟踪梯度 |
| `clamp` f_pred 位置 | **已删除** | 坏 world 由训练脚本过滤 |
| 动力学速度 clamp | ±30 m/s | 安全网 |
| Wheelbase L | **5.0 m** | 匹配 Waymo 实车 |
| Steering range | **±0.6 rad**（~34.4°） | min 转弯半径 ~6.8m |
| Actor acc bias | 0.05（tanh 后约 0.075 m/s²） | 防刹停 |
| f_pred 通道映射 | raw_steer×0.6→rad, raw_acc×1.5/3.0→m/s², stack([acc,steer]) | 与 select_action 一致 |
| delta_phi | **`ref_heading - ego_theta`** | 真正航向对齐误差 |
| 道路 penalty | `relu(edge_dist - 2.0)` 线性 | 始终有梯度 |
| 坏世界过滤 | 阶段1 预训练路径检查 + 每步 ego 检查 | 异常值不进 buffer |
| 候选路径选择 | **episode 级别固定** | 消除步间跳变 |
| 道路宽度 | **动态读取 segment_width** | invalid 时 fallback 3.75 |
| rho 模式 | `--init-penalty 0` = 纯追踪, `≥1.0` = 含 penalty | 替代旧的 `--tracking-only` flag |
| 世界重采样 | **坏世界 > `--max-bad-worlds` 时自动触发** | 从密集池随机抽取新世界 |
| 数据池大小 | **`--dataset-size` 0 = 自动检测** | 重采样候选池 |
| 参考路径 pos/heading | **expert** | 精确道路中心线 |
| 参考路径 speed | **`_curvature_speed` 曲率限速** | v_max=20m/s, a_lat=2.5m/s² |
| 密度缓存 | **`--min-partner-density 2.0`** | 随机抽样 2000 文件，30 秒生成 |
| 稠密池 | **`--dense-sample-size 500`** | resample 优先从此池采样 |

---

## 35. history_loss 重复积累导致损失图横坐标错误

**现象**：LossPlotter 画的 epoch 聚合图横坐标是 save 次数而非 epoch 数，step 图存在重复数据点。

**根因**：
- `history_loss = []` 在 epoch 循环外初始化，每 epoch append 后从不重置 → 累积所有 epoch 数据
- `agent.save()` 每次 save 时把**完整累积列表** append 到 `self.history_loss` → 数据反复嵌套
- `save_freq=5` 时，`self.history_loss` = `[[epoch1-5数据], [epoch1-10数据], ...]` → epoch 6-10 重复了 epoch 1-5

**修复**（3 处）：

| 文件 | 改动 |
|------|------|
| `idc-train.py` | `history_loss = []` 移到 epoch 循环内 → 每 epoch 独立 `epoch_history` |
| `idc-train.py` | epoch 结束：`agent.history_loss.append(epoch_history.copy())` |
| `idc_agent.py` | `save()` 中删除 `self.history_loss.append(save_info.get(...))` |

修复后 `self.history_loss = [[epoch1], [epoch2], ...]`，横坐标正确，无重复数据。

---

## 36. 评估脚本缺少世界过滤

**现象**：评估导出的 GIF 图像中部分车辆瞬移/飞出路外，画面不正常。

**根因**：`idc-eval.py` 没有任何世界过滤逻辑，而 `idc-train.py` 有两级过滤（路径坐标 + ego 坐标）。

**修复**（`idc-eval.py`）：

| 位置 | 改动 |
|------|------|
| `generate_candidate_paths` 后 | 阶段 1：路径坐标异常检测 → `bad_worlds` |
| 每步 `get_idc_observations_batch` 后 | 阶段 2：ego 坐标检查，异常追加到 `bad_worlds` |
| viz/recording | `viz_list` 只建好世界，位置记录和 GIF 帧跳过 `bad_worlds` |

---

## 37. 世界重采样：过滤后自动补新世界

**现象**：初始 200 个世界中 ~5% 路径异常 + 每 epoch ~1% ego 跳变 → 训练中好世界越来越少 → 过拟合剩余世界。

**方案**：当好世界数低于阈值时，从全量数据池随机抽取新世界替换。

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num-worlds` | 200 | 并行训练世界数（A4000 16GB 建议 200，32GB 可 300） |
| `--dataset-size` | 0（自动） | 全量候选池大小，0=检测 `data_dir` 下 tfrecord 文件数 |
| `--max-bad-worlds` | 100 | 坏世界数超过此值时触发重采样 |

### 流程

```
初始化：从全量池随机加载 200 个世界，过滤 → bad_worlds

每 epoch 5 步一次：[DIAG-ref] good_worlds=195/200

每 epoch 结束：
  if len(bad_worlds) > 100:
    → random.sample(全量池, 200)
    → env.swap_data_batch()  # 换入新世界
    → 重建 builder expert 数据 + 候选路径
    → 阶段 1 过滤
    → agent.update_ego_indices() + clear_buffer()
    → 下一 epoch 用新世界开始
```

### 涉及文件

| 文件 | 改动 |
|------|------|
| `idc-train.py` | 新增 `glob`/`random` 导入，CLI 参数，`all_files` 列表，`resample_worlds()` 函数（~45 行），epoch 结束触发 |
| `idc_agent.py` | 新增 `update_ego_indices()` + `clear_buffer()` |

### 注意

- swap 时 buffer 必须清空（旧 state 的 world_idx 映射到旧世界，temporal 不连续）
- buffer 容量 100,000 → 半分钟内填满，不影响训练连续性
- 默认 200 worlds ~11GB VRAM，300 worlds ~16.5GB → A4000 16GB 用 200

### State 布局（59 维）

| 区间 | 维度 | 内容 |
|------|------|------|
| [0:6] | 6 | ego |
| [6:38] | 32 | others |
| [38:46] | 8 | validity |
| [46:58] | 12 | ref_error: `[dp, dphi, dv, lat+tphi+troad ×3 at t+3/6/9]` |
| [58:59] | 1 | temporal |

### ref_error 细节（12 维）

| 索引 | 含义 | utility | penalty |
|------|------|:---:|:---:|
| [0] | `delta_p` 当前横向误差 | ✓ | — |
| [1] | `delta_phi` 当前航向误差 | ✓ | — |
| [2] | `delta_v` 当前速度误差 | ✓ | — |
| [3-5] | t+3: lat / dphi / road_dist | lat+dphi ✓ | 仅 state |
| [6-8] | t+6: lat / dphi / road_dist | lat+dphi ✓ | 仅 state |
| [9-11] | t+9: lat / dphi / road_dist | lat+dphi ✓ | 仅 state |

**road_dist 语义**：参考路径上方道路边界距离（路径属性，不受自车偏航影响）。penalty 仍用 `get_road_edges_batch` 实时计算自车位置的道路距离，两者独立。

### 统一框架改动

| 改动 | 说明 |
|------|------|
| 移除 `self.tracking_only` | 不再有 tracking/full 模式分支 |
| `if tracking_only: p = zeros` → 移除 | penalty 由 `self.rho * p` 控制，rho=0 时自然归零 |
| `if not tracking_only: noise_decay` → 移除 | 所有模式统一衰减 σ×0.98/PIM |
| `if not tracking_only: rho *= c` → 移除 | rho 始终放大，rho=0 时保持 0 |
| CLI: `--init-penalty 0` 替代 `--tracking-only` | 追踪模式：rho=0；正常：rho=1.0 |

### 性能优化：road_dist 预计算 + GPU ref_tensor

**问题**：每次 `f_pred_batch` 调用 `_road_dist_batch` 做 O(N) 最近邻搜索 → 每 rollout 3 万次搜索 → 训练极慢。

**修复分两步**：

**Step 1 — 预计算 road_dist**：`generate_candidate_paths` 中一次性算好 91×3 点的 road_dist，存入 `path['road_dist']`，后续 O(1) 查表。

**Step 2 — GPU ref_tensor**：初始化时构建 `self.ref_tensor` [num_worlds, 1, num_paths, 91, 5]，存 pos_x/pos_y/speed/heading/road_dist，放在 GPU。`get_ref_states_batch` 和 `get_road_dist_batch` 新增 GPU fast path，用 `ref_tensor[w, a, p, t]` 单次 tensor 索引替代 Python 循环。

| 接口 | 改前 | 改后 |
|------|------|------|
| `get_ref_states_batch` ×4 | 4×512 Python 浮点转换 + `torch.tensor` | 4 次 GPU tensor 索引 |
| `get_road_dist_batch` ×3 | O(N) 最近邻搜索 ×3 | 0 次（归入 ref_tensor 索引） |
| **每步总 Python→C 往返** | ~4000 次 | **~0 次** |

### 相关文件改动汇总

| 文件 | 方法/位置 | 改动 |
|------|----------|------|
| `idc_agent.py` | `__init__` | `DIM_REF_ERROR=12`, 移除 `tracking_only`, `noise_std=0.1` |
| `idc_agent.py` | `f_pred_batch` | 3 前瞻点 + `get_road_dist_batch`（GPU 路径） |
| `idc_agent.py` | `utility_batch` | 3 点 lat+dphi 成本，road 不入 utility |
| `idc_agent.py` | `update_actor` | 移除 tracking_only 分支 |
| `idc_agent.py` | `update` | 移除 rho 放大的 tracking_only 守卫 |
| `idc_state_builder.py` | `generate_candidate_paths` | road_dist 预计算 + 构建 `ref_tensor` |
| `idc_state_builder.py` | `get_ref_states_batch` | GPU fast path（当 temporal_indices 传入时） |
| `idc_state_builder.py` | `get_road_dist_batch` | GPU fast path（`ref_tensor[..., 4]`） |
| `idc_state_builder.py` | `get_idc_observations_batch` | 3 前瞻点 road_dist 直接查 `path['road_dist']` |
| `idc_state_builder.py` | 新增 `_road_dist_point` | @staticmethod 道路距离纯函数 |
| `idc_state_builder.py` | 删除 `_road_edge_dist` | 被 `_road_dist_point` + 预计算替代 |
| `idc_agent.py` | 删除 `_road_dist_batch` | 被 builder 的 `get_road_dist_batch` 替代 |
| `idc-train.py` | CLI args | 删除 `--tracking-only`, `--init-penalty` 默认 0 |
| `idc-train.py` | DIAG-init | 修复 `get_ref_state_from_path` bug（直接读 candidate_paths） |

### 训练命令

```bash
# 纯追踪（rho=0）:
python src/scripts/train/idc-train.py --init-penalty 0 --data-dir <path>

# 正常训练（rho=1.0）:
python src/scripts/train/idc-train.py --init-penalty 1.0 --data-dir <path>
```

---

## 38. Bezier 弧线偏离 → Expert 轨迹替代

**现象**：Bezier 参考路径在弯道上产生 1-3m 弧线偏差，车辆 tracking 差、参考路径"开出路外"。

**根因**：Cubic Bezier 仅约束起终点航向，中间 80m 自由发挥。起终点航向差异越大，弧线偏离越严重。

**尝试过的方案**：

| 方案 | 效果 | 结果 |
|------|------|------|
| 多段 Bezier（直线锚点） | 直线锚点在弯道外，弧线反而更大 | ✗ 放弃 |
| 多段 Bezier（圆弧锚点） | 数学正确但 Bezier 在急弯处会跨过道路边界 | ✗ 放弃 |

**最终方案**：训练时直接用 expert 轨迹（零偏差），Bezier 保留供 CARLA 迁移。

### 实现

`generate_candidate_paths` 中路径生成：

```
pid=1（中心线）: expert_pos 直接复制 91 点
pid=0（左偏移）: expert_pos 每点沿左垂向偏移 -lane_width
pid=2（右偏移）: expert_pos 每点沿右垂向偏移 +lane_width

heading → expert_heading 直接复制
speed → expert_vel 直接复制
road_dist → 预计算不变
```

`_make_multi_bezier_path` 代码保留（不调用），CARLA 迁移时启用。

### 涉及文件

| 文件 | 改动 |
|------|------|
| `idc_state_builder.py:54-103` | `generate_candidate_paths` 中 Bezier → expert 逐点偏移 |
| `idc_state_builder.py:129-226` | `_make_multi_bezier_path` 保留（不删） |

---

## 39. 评估脚本支持 rho 参数覆盖

**问题**：`agent.load()` 会从 checkpoint 恢复 `self.rho`，无法在评估时用 CLI 覆盖。

**修复**：`idc-eval.py` 中 `agent.load()` 后新增：

```python
agent.rho = args.init_penalty
agent.max_penalty = args.max_penalty
```

现在可以：

```bash
# 纯追踪评估
python idc-eval.py --init-penalty 0 --model-path ...

# 含 penalty 评估
python idc-eval.py --init-penalty 1.0 --max-penalty 10.0 --model-path ...
```

---

## 40. 世界重采样 OOM 修复

**现象**：`swap_data_batch` 时 CUDA OOM（CUDA_ERROR_OUT_OF_MEMORY）。

**根因**：swap 前 builder 持有旧世界的 `ref_tensor`、`expert_pos/vel/heading` 等 GPU 张量。`sim.set_maps()` 加载新世界时新旧数据叠加 → 双倍显存 → OOM。

**修复**：`resample_worlds` 开头主动释放：

```python
for attr in ['ref_tensor', 'expert_pos', 'expert_vel', 'expert_heading', 'candidate_paths', '_road_cache']:
    if hasattr(builder, attr):
        delattr(builder, attr)
torch.cuda.empty_cache()
```

然后 `_setup_expert_data()` + `generate_candidate_paths()` 重建新数据。

---

## 41. GPUDrive 坐标异常确认

**现象**：训练中每 epoch 过滤 50+ 世界，远远超过预期的 1-2/ep。

**诊断**：检查 `[FILTER-ego]` 日志，所有被过滤世界的 ego 坐标均为 `(-11000, -11000)`。这是 GPUDrive 模拟器的已知缺陷——某些场景在 episode 中触发坐标跳变。

**结论**：不是 IDC 代码问题，是世界重采样阈值方案正确运行的表现。`bad_worlds` 统计和 threshold 机制正常工作。

---

## 42. Road type 语义澄清

**现象**：可视化图中 road_line（灰色）看起来像车道边缘，road_edge（棕色）像中心线。

**GPUDrive 原始 type 映射**：

| Type 值 | 名称 | 含义 |
|:---:|------|------|
| 1 | road_line | 车道标线（虚线/实线） |
| 2 | road_edge | 道路边界（路沿/中间带） |
| 3 | lane_center | 车道中心线 |

当前代码 `_road_dist_point` 使用 `type=1`（road_line = 车道标线），计算"到最近车道标线的距离"。

| 模块 | type=1 效果 | type=2 效果 |
|------|-----------|-----------|
| road_dist（前瞻状态） | 到最近车道标线距离 | 到道路边界距离 |
| road_violation（penalty） | 偏离标线 > 2m 触发 | 偏离边界 > 2m 触发 |

**结论**：tracking-only 阶段（rho=0）无影响。penalty 阶段（rho>0）时再讨论是否切换 type。

---

## 43. eval 脚本 CLI 更新

评估脚本 `idc-eval.py` CLI 已支持：

```bash
--init-penalty 0        # 纯追踪评估
--init-penalty 1.0      # 含 penalty 评估
--max-penalty 10.0      # ρ 上限
```

---

## 44. 路径生成终版 + 密度采样 + 统一框架（大清理）

**日期**：2026-05-26

**目标**：统一默认路径生成方式为 expert pos/heading + 曲率限速，删除废弃代码，加入密度缓存降低避障场景稀疏问题。

### 路径生成：Expert + Curvature Speed（唯一默认）

删除 3 个废弃方法，统一默认：

| 删除 | 原因 |
|------|------|
| `_make_bezier_path`（单段 Bezier） | 弧线偏差 ~1.5m |
| `_make_multi_bezier_path`（多段 Bezier） | 圆弧锚点方案也放弃了 |
| `_make_spline_path`（lane_center spline） | spline 在交叉口 road ID 断裂 |

保留 `_curvature_speed`（曲率限速）：

```
v[i] = min(20, sqrt(2.5 / (|κ[i]| + 1e-6)))
v_max = 20 m/s (72 km/h), a_lat_max = 2.5 m/s²
```

3 条候选路径（pid 0/1/2）：
```
pos    → expert_pos + lateral_offset（逐点垂向偏移）
heading → expert_heading（直接复制）
speed  → _curvature_speed(pos)（纯静态，消除避障减速信号）
```

删除 `--use-road-spline` CLI flag，net 净删除 ~175 行。

### 密度缓存 + 稠密世界采样

**问题**：Waymo 数据稀疏场景多，penalty 训练时触发信号不足。

**方案**：从全量池随机抽样 2000 文件计算 partner density，缓存到 JSON。

```python
# 密度检测单次 ~30 秒
probe_files = random.sample(all_files, 2000)
for each batch in probe_files:
    valid partner count → density_cache[file] = avg_count

# 训练中 resample 时优先稠密池
dense_files = sorted by density[:dense_sample_size]
batch = random.sample(dense_files, num_worlds)  # 稠密池够大时
```

| CLI 参数 | 默认 | 说明 |
|------|--------|------|
| `--min-partner-density` | 2.0 | 平均周车数阈值 |
| `--dense-sample-size` | 500 | 稠密候选池大小 |

缓存独立于 checkpoint，依赖不变可复用。

### 其他本次改动

| 改动 | 说明 |
|------|------|
| 路径生成简化 | 无分支、无 CLI flag、统一默认 |
| eval CLI rho 覆盖 | `agent.load()` 后 `agent.rho = args.init_penalty` |
| 密度缓存用主 env | 不再创建第二个 GPUDriveTorchEnv（避免 setCudaHeapSize 冲突） |
| `os.listdir` 删除 | 70k 文件列表不再打印到 stdout |

### 删除的废弃方法清单

| 方法 | 文件 | 行 |
|------|------|-----|
| `_make_bezier_path` | idc_state_builder.py | ~37 |
| `_make_multi_bezier_path` | idc_state_builder.py | ~110 |
| `_make_spline_path` | idc_state_builder.py | ~77 |
| `_road_edge_dist` | idc_state_builder.py | ~13 |
| `_road_dist_batch` | idc_agent.py | 3 |

---

## 45. NaN + 跟踪崩溃诊断与修复

**日期**：2026-05-27

**症状**：epoch 428 时 gep_iter=1297，rho=20（cap），actor_loss=1262，Actor 权重 NaN。82/150 worlds 被标记 bad（坐标偏离 >200m），max pos_err=31m，delta_phi≈1.090 rad（~62°）。

**根因**：三条原因形成恶性循环：

1. **rho 无条件放大**：每个 PIM 周期（每 30 PEV）无条件乘 amplifier_c，无论是否真有碰撞/越界。导致 rho 必然抵达 cap=20
2. **Penalty 无上限**：单步 `road_violation` 可达数千（ego 偏离道路时 `edge_dist` 巨大），与 rho=20 相乘 → 单步 cost 可达 100k
3. **梯度爆炸**：超高 penalty 梯度累积在 Adam 的二阶矩中，最终导致 NaN
4. **Actor 初始偏置为零**：tanh 输出 ~0 → 初始动作≈无控制，ego 滑行偏离

**修复（4 项）**：

| 修改 | 文件 | 效果 |
|------|------|------|
| Actor 输出层 `bias[0]=0.05` | `continuous_actor_critic.py` | 初始动作偏轻加速，防静止滑行偏离 |
| `rho` 仅在检测到 violation 时放大 | `idc_agent.py:update()` | 无碰撞/越界时 ρ 保持不变 |
| `p = torch.clamp(p, max=100.0)` | `idc_agent.py:update_actor()` | 单步 penalty 上限，防极端偏离引爆梯度 |
| 过滤阈值 5000→200 | `world_manager.py` | 坐标 200 已远超 GPUDrive ±100 边界，减小 marginal steeper penalty |

**流程变更**：`update_actor()` 返回值从 `actor_loss` 改为 `(actor_loss, has_violation)`。`has_violation = max_penalty > 0.01`（rollout 中任意 step 有任何 penalty > 0.01 即判定有违规）。NaN 回滚仅在 `has_violation=True` 时递减 gep_iter/rho。

---

## 46. Resample 追踪模式使用全量池

**日期**：2026-05-27

**问题**：`--init-penalty 0` 纯追踪模式下，`WorldManager.resample()` 仍优先从 `dense_files`（稠密场景池）抽取，限制了道路类型多样性。

**修复**：`world_manager.py:resample()` 增加 `agent.rho > 0` 判断：

```python
candidate_pool = dense_files if agent.rho > 0 and len(dense_files) >= num_worlds * 2 else all_files
```

rho=0 时自动走全量池，rho>0 时优先稠密池。无需额外 CLI flag。

---

## 47. YAML 配置系统

**日期**：2026-05-27

**目标**：将所有硬编码参数和 argparse 默认值迁移到 YAML 配置文件，CLI 仅保留覆盖项。

**新增文件**：

| 文件 | 说明 |
|------|------|
| `configs/default.yaml` | 47 个参数的完整默认配置，按 `env`/`training`/`agent`/`dynamics`/`world`/`paths`/`diag` 分组 |
| `src/utils/config.py` | `build_config(path, overrides)` — YAML 加载 → 扁平化 → CLI 覆盖 → `SimpleNamespace` |

**修改文件**：

| 文件 | 变更 |
|------|------|
| `idc_agent.py` | `args → config`：所有硬编码值（cost weights、noise、safety distances、gamma、buffer_capacity）从 config 读取；`KinematicBicycleModel` 接收 `wheelbase/lr_ratio/v_max` |
| `world_manager.py` | `filter_threshold` 从 `config.filter_threshold` 读取 |
| `idc-train.py` | argparse 从 30 个参数精简到 20 个覆盖项，`train(args)` → `train(config)` |
| `idc-eval.py` | 同上，`evaluate(args)` → `evaluate(config)` |

**扁平化规则**：YAML 嵌套结构按层级展开，一级 key 丢弃，二/三级 key 作为最终 key：

```yaml
training:
  dt: 0.1      → config.dt = 0.1
agent:
  noise:
    std: 0.1   → config.noise_std = 0.1
```

**CLI 覆盖**：CLI `--key value` 中 value 非 None 时覆盖 YAML 对应值。

---

## 48. 代码重构：WorldManager + env_utils

**日期**：2026-05-27

**目标**：将 `idc-train.py` 中的密度缓存、世界过滤、重采样逻辑抽象为可复用的 `WorldManager` 类，train/eval 共用。

**新增模块**：

| 文件 | 行数 | 职责 |
|------|------|------|
| `src/env/__init__.py` | 10 | 导出 |
| `src/env/env_utils.py` | 44 | `extend_action_to_3d`, `get_env_config`, `load_scenes`, `get_ego_indices` |
| `src/env/world_manager.py` | 190 | `WorldManager`：密度缓存 + 过滤（路径/ego 异常）+ 重采样 + DIAG 日志 |

**脚本精简**：

| 脚本 | 之前 | 之后 | 减少 |
|------|------|------|------|
| `idc-train.py` | 451 行 | 当前（YAML + WorldManager） | ~57% |
| `idc-eval.py` | 228 行 | 当前（YAML + WorldManager） | ~32% |

**WorldManager API**：

```python
wm = WorldManager(env, builder, agent, all_files, args, logger, compute_density=True)
wm.filter_initial(ego_indices)           # 阶段1：路径坐标异常过滤
wm.filter_per_step(states, step)         # 阶段2：ego 坐标 + DIAG ref 误差
wm.should_resample()                     # bad_worlds > max_bad_worlds?
ego_indices = wm.resample()              # swap + 重建 + 过滤 + agent 更新
wm.good_worlds                           # 未过滤的 world 列表
wm.good_count                            # 存活 world 数量
```

---

## 49. Road Penalty 公式反向（CRITICAL）

**日期**：2026-05-28

**症状**：penalty 训练后 tracking 反而更差（good_worlds 119→34），actor_loss 被 penalty 主导（~800 vs tracking ~20）。LA-DIAG 显示 delta_phi≈0.034（heading 完美）但 delta_p=8m（偏移很大），road=7.5m。

**根因**：`penalty_batch` 中 road_violation 公式为：

```python
edge_dist = sqrt((edge_pts - ego)^2)       # 自车到最近道路线距离
road_violation = F.relu(edge_dist - D_road_safe)  # ← 错误
```

含义：
- 路中心（edge_dist=7.5m）→ violation = 5.5，**被罚**
- 贴路边（edge_dist=0.5m）→ violation = 0，**奖励**

penalty 把自车推向路边，tracking 拉向路中心 → 冲突。rho=10 时 penalty 梯度（55/步）是 tracking 梯度（4.8/步）的 11 倍，Actor 被推向路边。

**修复**：改为 path-based 公式，利用已预计算的 `road_dist_ref`（参考点到最近路边）和 state 中的 lateral error `lat`：

```python
ego_road_dist = road_dist_ref - torch.abs(lat)
road_violation = F.relu(D_road_safe - ego_road_dist)
```

| 场景 | lat | road_dist_ref | ego_road_dist | violation |
|------|-----|-------------|-------------|-----------|
| 路中心 | 0 | 7.5 | 7.5 | **0** ✓ |
| 靠路边 | 5.5 | 7.5 | 2.0 | 0 ✓ |
| 跨出路面 | 8 | 7.5 | -0.5 | **2.5** ✓ |

**涉及文件**：`idc_agent.py:460-505`（penalty_batch 签名增加 `p_i` 参数）

---

## 50. 权重不平衡：heading 信号被 position 淹没

**日期**：2026-05-28

**症状**：episode 内 pos_err 从 1m 递增到 31m，critic loss=2863，tracking 不稳定。

**根因**：utility 函数中 `pos_err²×0.3` 最大可达 750，`heading_err²×0.3` 最大仅 3，比例 250:1。heading 对 utility 的贡献在数值上可忽略，Actor 梯度几乎全部来自 position error。但减小 position error 需要先改正 heading。

**修复**：`pos_err_weight: 0.3 → 0.03`（/10），比例降为 25:1。同时 `lr_critic: 3e-4 → 1e-4`（value 跨度大时拟合更稳）、`noise_decay_rate: 0.98 → 0.95`（更早收敛到确定性策略）。

**涉及文件**：`configs/base.yaml`

---

## 51. Checkpoint 续训时 optimizer lr 被旧值覆盖

**日期**：2026-05-28

**症状**：修改 `base.yaml` 中 `lr_critic=1e-4` 后加载 tracking checkpoint 续训，critic 仍用 3e-4。

**根因**：`load_checkpoint()` → `opt.load_state_dict()` 恢复 optimizer 完整状态，包括 `param_groups['lr']`。旧 checkpoint 中 lr=3e-4 覆盖了新建 optimizer 时设置的 1e-4。rho 同理，但 rho 有专门处理逻辑（checkpoint rho=0+init_penalty>0 时从 config 取值）。

**修复**：`load()` 中 `load_checkpoint()` 之后显式覆写：

```python
for pg in self.actor_optimizer.param_groups:
    pg['lr'] = self.config.lr_actor
for pg in self.critic_optimizer.param_groups:
    pg['lr'] = self.config.lr_critic
```

rho 不受影响（保持现有逻辑），max_penalty 在 agent 构造时从 config 读也不受影响。

**涉及文件**：`idc_agent.py:load()`

---

## 52. GIF fps 与录制间隔混淆

**日期**：2026-05-28

**症状**：用户调低 `gif_fps=1` 期望更流畅播放，实际 GIF 变成每秒只播放 1 帧，又慢又卡。

**根因**：`gif_fps` 同时控制 VisualRecorder 的录制间隔（`step % fps` 决定何时录帧）和 GIF 播放速度（`imageio.mimsave(fps=fps)`）。降低 fps → 录制间隔变大 + 播放更慢 → 双重恶化。

**修复**：解耦为两个独立参数：

| 参数 | 默认 | 控制 |
|------|------|------|
| `gif_record_interval` | 2 | 每隔 N 个 env step 录一帧 |
| `gif_fps` | 15 | GIF 播放速度（帧/秒） |

录制间隔用 `config.gif_record_interval`，播放用 `custom_fps=config.gif_fps`。

**涉及文件**：`configs/train.yaml`、`configs/eval.yaml`、`idc-eval.py`

---

## 53. Rho 乘性膨胀 → 线性递增（统一训练框架）

**日期**：2026-05-28

**症状**：`rho *= amplifier_c`（乘性膨胀）每 30 步无条件加倍率，必然到 cap → penalty 压倒 tracking → 需分离 tracking/penalty 阶段训练。用户反馈分离训练繁琐，希望统一流程。

**根因**：乘性膨胀几何增长，违规率不管高低 rho 都会到 cap。跟踪阶段的 rho=0 和 penalty 阶段的 rho>0 切换不自然。

**修复**：改为线性递增 `rho += amplifier_c`，仅在检测到违规时增加。

| 参数 | 旧值 | 新值 | 语义 |
|------|------|------|------|
| `amplifier_c` | 1.005（乘性倍率） | 0.01（线性增量） | 每违规 PIM +0.01 |
| `init_penalty` | 0.0 | 0.01 | 起点几乎为 0 |
| `max_penalty` | 5.0 | 3.0 | 更低 cap 防压制 tracking |

演进曲线：epoch 0 rho=0.01（tracking）→ epoch 50 rho≈1.2 → epoch 100 rho≈3.0 cap（平衡态）。

NaN 回滚同步改为 `rho = max(0.0, rho - amplifier_c)`。

**涉及文件**：`base.yaml`、`idc_agent.py:update()`、`idc_agent.py:NaN rollback`

---

## 54. PDMS 评估系统

**日期**：2026-05-28

**目标**：集成 CARLA 风格的 PDMS（Planning Decision Making Score）到训练和评估。

**公式**：`PDMS = (NC × DAC × DDC) × (EP×5 + TTC×5 + C×2 + LK×2) / 14`

**新增模块**：

| 文件 | 职责 |
|------|------|
| `src/metrics/__init__.py` | 导出 |
| `src/metrics/pdms.py` | `PDMSScorer`（在线累加器）+ `RolloutPDMSScorer`（评估前向推演） |
| `src/metrics/plotter.py` | `print_pdms_table`（ASCII 表格）+ `plot_pdms_radar`（雷达图）+ `plot_pdms_bar`（柱状图） |

**数据来源**：GPUDrive `info_tensor`（off-road/collision）、`partner_observations_tensor`（TTC）、`self_observation_tensor`（speed/jerk）、state ref_error（lat/delta_phi）。

**训练集成**：每 epoch `[PDMS] score=72.3 completion=89.2% collisions=3 off_road=12`

**评估集成**：终端表格 + `pdms_plots/` 图表 + step 0 推演 Rollout PDMS 预测

**涉及文件**：`src/metrics/*`、`base.yaml`、`idc-train.py`、`idc-eval.py`、`idc_agent.py`

---

## 55. 密度离线扫描 + 范围选择

**日期**：2026-05-28

**问题**：训练时按周车密度筛选场景需反复在线扫描，慢（~30s 抽样）且不支持精确范围。

**新增**：`src/scripts/utils/scan_density.py` — 全量扫描所有 tfrecord 的周车密度，输出 JSON。

**WorldManager 增强**：
- `density_cache_file`：直接加载外部全量 JSON（跳过在线扫描）
- `min_partner_density` / `max_partner_density`：密度区间筛选
- `dense_sample_size=0`：不限候选池大小

**Waymo 70k 实际分布**：均值 19.19，中位数 15，0-63 范围。21% 场景密度 ≥31。

**涉及文件**：`scan_density.py`、`world_manager.py`、`base.yaml`、`train.yaml`、`eval.yaml`

---

## 56. 诊断 penalty flag

**日期**：2026-05-28

**目标**：快速定位 road penalty vs vehicle penalty 哪个是问题源。

**新增 CLI**：`--no-road-penalty` / `--no-veh-penalty`（均 `action='store_true'`）

**实现**（`idc_agent.py:penalty_batch()`）：
```python
if getattr(self.config, 'no_veh_penalty', False):
    veh_violation = torch.zeros_like(veh_violation)
if getattr(self.config, 'no_road_penalty', False):
    road_violation = torch.zeros_like(road_violation)
```

零侵入原有逻辑，仅将对应 violation 置零。

**使用**：
```bash
# 仅道路约束
python idc-train.py --no-veh-penalty
# 仅周车约束
python idc-train.py --no-road-penalty
```

**涉及文件**：`base.yaml`、`idc_agent.py`、`idc-train.py`、`idc-eval.py`

---

## 57. Expert 轨迹含 sentinel 坐标 → 参考点污染

**日期**：2026-05-29

**症状**：短轨迹 world 在 step 57 后自车偏离到 sentinel `(-11000, -11000)`，导致 filter 误标 bad。`pos_err=15585m` 持续污染 DIAG-ref 和 buffer。

**根因**：GPUDrive C++ 源码确认 `kPaddingPosition = (-11000, -11000)`（`src/consts.hpp:64`）。Waymo 数据中很多场景只有 50-70 步有效轨迹，expert data 在无效步填入 sentinel。`generate_candidate_paths` 将此 sentinel 作为参考点，自车在有效步结束后跟踪到 sentinel → 动作偏离。

**修复**：`idc_state_builder.py:generate_candidate_paths()` 中在构建路径前检测 expert data 中的 sentinel（`abs(pos) > 5000`），找到截止点后用最后有效位置/heading 填充剩余步。所有下游消费者（ref_tensor、get_ref_states_batch、get_road_dist_batch、PDMS 等）自动受益。

**涉及文件**：`idc_state_builder.py`

---

## 58. 到达终点的世界误标 bad → reached_worlds 分类

**日期**：2026-05-29

**症状**：good_worlds 从 150 快速降至 30-50，但多数世界跟踪质量良好（pos_err < 5m）；日志无 `[FILTER-ego]` 警告，DIAG-ref 却在持续。到达终点的世界被 sentinel 误标。

**根因**：GPUDrive 在 agent 到达目标（`reachedGoal=1`）或 episode 结束时触发 `done.v=1` → `movementSystem` 将 ego 移至 sentinel `(-11000, -11000)`（`src/sim.cpp:327-331`）。filter 检测到 sentinel 但无法区分是正常终点还是真实崩溃。

**修复**：引入三类世界分类：

| 集合 | 含义 | 生命周期 |
|------|------|---------|
| `bad_worlds` | 真实崩溃 | resample 时清 |
| `reached_worlds` | 正常到达终点 | **每 epoch 清** |
| `good_worlds` | 两者都不在 | 每 epoch 动态变化 |

- `filter_per_step`：sentinel 检测后用 `env.get_dones()` 替代 `info_tensor[:,:,3]` 判断，done=1 → `reached_worlds.add`
- epoch 开头：`env.reset()` 后立即拉 `get_dones()` 预检测已 done world
- `good_worlds` 属性：排除 both bad + reached
- buffer 插入、PDMS 采集、no-sign 处理全部改为使用 `good_worlds`
- `should_resample`：新增 `good_count < num_worlds * 0.3` 触发

**涉及文件**：`world_manager.py`、`idc-train.py`、`idc-eval.py`

---

## 59. 日志分层：区分正常终点 vs 崩溃

**日期**：2026-05-29

**症状**：无法从日志判断 good_worlds 下降是因为到达终点还是真实崩溃。

**修复**：

- `[REACHED-GOAL] world_X step=35/91 — done, excluding`（正常终点，info 级别）
- `[FILTER-ego] world_X step=30/91 ego=(-11000,-11000)`（真实崩溃，warning 级别）
- `[DIAG-ref] good=135 reached=7 bad=8/150`（一目了然）

行人一眼能看到 150 个世界中：135 个正常训练、7 个到达终点、8 个真实崩溃。

**涉及文件**：`world_manager.py`

---

## 60. Route Completion 用固定 91 作分母 → PDMS 分数虚低

**日期**：2026-05-29

**症状**：模型完美追踪到终点，但 PDMS 显示 `completion=56.3%`。短轨迹世界（35 步）到达终点后 completion = 35/91 = 38%。

**根因**：`PDMSScorer.update_step` 的 `max_step` 参数传入固定 `config.max_step=91`，而非该世界的实际候选路径长度。

**修复**：train/eval PDMS 采集处改为 `max_step=len(path['pos'])`，使用各世界 sentinel 裁剪后的实际路径长度。

**涉及文件**：`idc-train.py`、`idc-eval.py`

---

## 61. PDMS 表格缺分数分布 → 看不出是少数世界拖垮还是普遍问题

**日期**：2026-05-29

**问题**：PDMS 终端表格只显示均值和违规数，无法一眼判断 150 个世界中多少世界得分 >80、多少挂掉。

**修复**：`print_pdms_table` 新增一行分数分布：

```
Score distribution:
  ≥80: 92  |  60-79: 35  |  30-59: 18  |  <30: 5
```

**涉及文件**：`metrics/plotter.py`

---

## 62. Resample Interval 优化 → 提升数据多样性

**日期**：2026-05-29

**问题**：`resample_interval=50` → 400 epoch 仅见过 ~1500 个世界 → 70k Waymo 数据几乎未用。

**修复**：`resample_interval: 50 → 3`。每 3 epoch 换 150 个新世界，400 epoch 覆盖 ~20,000 个场景。buffer 清空成本 < 1.5%（填充仅需 4 步）。

**涉及文件**：`base.yaml`

---

## 63. Expert 轨迹可视化仍含 sentinel → 图像飞出地图

**日期**：2026-05-29

**症状**：Issue 57 的 sentinel 裁剪只修改了本地 numpy 副本，原始 `self.expert_pos` tensor 未被更新。`traj_visualizer.py` 直接读取 `self.expert_pos` → 短轨迹世界的 sentinel 尾段仍在图中显示为飞出地图的线。

**修复**：`generate_candidate_paths` 裁剪后将修正值写回原始 tensor。

**涉及文件**：`idc_state_builder.py`

---

## 64. 弯道追踪差 — 前瞻步子太短

**日期**：2026-05-29

**症状**：直行效果好但弯道差。20m/s 进弯需要 24m 刹车距离，当前 t+9 只有 0.9s/18m 提前量 → Actor 看到弯道时已来不及减速。

**根因**：前瞻步子 t+3/6/9 过短。弯道速度信号从 20→8 m/s 的下降在 t+9 才开始，物理上刹车距离不够。

**修复**：

| 位置 | 旧 | 新 |
|------|-----|-----|
| `get_idc_observations_batch` | t+3/6/9 | t+5/10/15 |
| `f_pred_batch` | temporal_next+3/6/9 | +5/10/15 |
| `speed_err_weight` | 0.1 | 0.3 |
| `acc_cost_weight` | 0.005 | 0.01 |

state 维度不变（62），旧 checkpoint 可直接加载续训。

**涉及文件**：`idc_state_builder.py`、`idc_agent.py`、`base.yaml`

---

## 65. 训练速度优化

**日期**：2026-05-29

**目标**：减小每步训练循环的冗余开销。

**修复项**：

| # | 优化 | 文件 | 效果 |
|---|------|------|------|
| 1 | `abs/rel/partner` 每步拉一次，state builder + positions 复用 | `idc_state_builder.py`、`idc-train.py` | 省 2 次 GPU→CPU/步 |
| 2 | `good_worlds` 属性加缓存（dirty flag） | `world_manager.py` | 省 2-3 次 O(N) 列表重建/步 |
| 3 | `f_pred_batch` 末尾 `torch.cat` 替代 Python for-loop | `idc_agent.py` | 省 rollout 内 Python 开销 |
| 4 | `no_sign`/`buffer`/`increment` 三循环合并 | `idc-train.py` | 省 300+ 次迭代/步 |

累计 **~20% 提速**，所有改动零逻辑变更。

**涉及文件**：`idc_state_builder.py`、`world_manager.py`、`idc_agent.py`、`idc-train.py`

---

## 66. 余弦退火 LR 调度器

**日期**：2026-06-01

**目标**：固定 LR（8e-5）初期收敛慢、末期震荡大 → 引入 CosineAnnealingLR。

**修复**：用 PyTorch 内置 `CosineAnnealingLR`，T_max 根据 epochs/batch_size 自动估算。

| 参数 | 值 | 说明 |
|------|-----|------|
| `lr_actor_max` | 1e-4 | cosine 起点 |
| `lr_actor_min` | 1e-6 | cosine 终点 |
| `lr_critic_max` | 3e-4 | |
| `lr_critic_min` | 1e-5 | |

checkpoint 保存/加载 `last_epoch`，续训无缝衔接。删除旧的 `lr_actor`/`lr_critic` 参数。

**涉及文件**：`idc_agent.py`、`base.yaml`

---

## 67. Road width 异常 clamp + Actor tanh 移除

**日期**：2026-06-01

**症状**：
1. 某些世界候选路径飞出道路 → `segment_width` 异常大 (50-100m) → 候选路径 ±50m 偏移
2. DIAG-act 显示 steer 永久 ±0.6 rad（69% 饱和） → tanh 输出 ±1.0 不变

**根因**：
1. `dynamic_width` 从 nearest road point 读取，sentinel 世界的 nearest 点可能异常远 → 宽度值异常
2. tanh 输出饱和后梯度 ≈0 → 网络无法学习中间 steer 值

**修复**：
1. `dynamic_width` clamp 上限 15m，超过则 fallback 为 lane_width=3.75
2. Actor 输出移除 tanh，steer 改为 `clamp(raw*0.3, -0.6, 0.6)`（全程有梯度）、acc 保留 tanh

**涉及文件**：`idc_state_builder.py`、`continuous_actor_critic.py`、`idc_agent.py`

---

## 68. Steer bias → 0 + Acc bias → 1.2（低速→高速 regime）

**日期**：2026-06-01

**症状**：tanh 移除后 steer 仍 69% 饱和（DIAG-act 统计 22k 次 0.6）。ego 持续在 5-10 m/s 低速 → max steer 0.6 是低速下过弯的唯一解 → 恶性循环。

**根因**：bias[1]=0（acc 默认 0 m/s²）→ ego 29 步才能到 20 m/s → 80% 时间低速 → Actor 在"低速+max steer=唯一解"regime 学习。

**修复**：

| bias | 旧 | 新 | 效果 |
|------|-----|-----|------|
| `bias[0]` (steer) | 0.5 | 0.0 | steer 默认居中 |
| `bias[1]` (acc) | ~0 | 1.2 | acc=1.26 m/s²，16 步到 20 m/s |

ego 在 20 m/s 下 0.25 rad 转向足以过弯 → 网络自然学会不需要 max steer。

**涉及文件**：`continuous_actor_critic.py`

---

## 69. steer_cost_weight + Actor 输出线性化

**日期**：2026-06-01

**背景**：多轮调优发现 steer 行为主要由两项主导——steer_cost 权重和输出层激活函数。最终方案：
- `steer_cost_weight: 0.06`（平衡 0.3→0.03→0.06 迭代确定）
- Actor 移除 tanh，改为 `clamp(raw*0.3, -0.6, 0.6)` 线性输出（配合 BC loss）
- `bias[1]=1.2` 加速起点

| 阶段 | steer_cost | 效果 |
|------|-----------|------|
| 初始 | 0.3 | 转向被压制 |
| 降 | 0.03 | 全量转向，饱和 |
| 最终 | 0.06 | 配合 BC loss 达到平衡 |

**涉及文件**：`base.yaml`、`continuous_actor_critic.py`、`idc_agent.py`

---

## 70. Steer bang-bang 饱和 — 最终由 BC loss 解决

**日期**：2026-06-01 → 最终修：2026-06-02

**症状**：多轮参数调优后，DIAG-act 统计始终 97% 的 steer 在 ±0.6 rad。PDMS ~55 不上升。

**尝试历程**：steer_cost 降 30×、tanh 移除、bias 调整、高噪声探索——均无效。RL 探索机制无法逃离"max steer = 最优"的局部最低点。

**最终修复**：Issue 71/72 的 BC loss（`atan(L × Δh/Δs)` 计算 expert steer）打破饱和——epoch 280 时 steer 分布已全部转为 0.02-0.34 中间值。

**涉及文件**：`idc_state_builder.py`、`idc_agent.py`、`base.yaml`

---

## 71. 专家行为克隆 BC loss（方案 B）

**日期**：2026-06-02

**目标**：用 Waymo expert 轨迹中的转向信息作为监督信号，打破 bang-bang steer 饱和。

**实现**：

| 文件 | 改动 |
|------|------|
| `idc_state_builder.py` | `get_expert_steer_batch()`：从 expert heading/pos 差分计算近似 steer：`δ = atan(L × Δh / Δs)`，L=5.0 |
| `idc_agent.py:update_actor` | BC loss: `MSE(clamp(raw[0]*0.3,-0.6,0.6), expert_steer) × bc_weight` |
| `base.yaml` | `bc_weight: 0.1`、`noise_std: 0.1`、`noise_decay_rate: 0.95` |

BC 仅训练时生效——GEP rollout 仍是主 loss，BC 是辅助梯度信号。推理/评估时不需要 expert 数据。

**效果**（epoch 280）：steer 分布从 97% ±0.600 转为全部 0.02-0.34 中间值，bang-bang 饱和彻底打破。PDMS 从 ~50 开始回升。

**涉及文件**：`idc_state_builder.py`、`idc_agent.py`、`base.yaml`

---

## 72. BC 初始方案失败 — expert_actions 全为零

**日期**：2026-06-02

**症状**：BC loss 的 `expert_mean` 持续为 0。DIAG-bc 打印 `expert_mean=0.000` 全部 15 条。1000 epoch 后 steer 分布不变。

**根因**：`expert_trajectory_tensor()` 的 `6*T:16*T` 区间（预留给 expert actions）在 GPUDrive 中全部为 0——C++ 端预留了槽位但未填充真实数据。BC 从未学到非零的 expert steer。

**诊断**：打印 3 个 world×2 agent×3 step 的所有 10 个 action 索引，全部为 0.000。

**修复**：改用 `atan(L × Δheading / Δs)` 从已有 expert heading/pos 数据计算近似 steer。

**涉及文件**：`idc_state_builder.py`

---

## 73. 训练协议 + 候选路径终点修复

**日期**：2026-06-02

**问题 1**：3-epoch resample 导致 buffer 仅填 41k/100k，每个世界只练 3 轮，优先级采样无意义。Critic loss 在 ~300 epoch 后 plateau。

**问题 2**：偏移候选路径（pid=0/2）的终点 = expert 终点 ±3.75m → GPUDrive goal 判定（2m 阈值）永不触发 → offset 路径世界永远不会 reached

**修复**：

| 参数 | 旧 | 新 | 效果 |
|------|-----|-----|------|
| `buffer_capacity` | 100k | 200k | 10 epoch 填满 |
| `batch_size` | 512 | 256 | 小 batch 减少极端 utility 污染 |
| `horizon` | 20 | 30 | 更长前瞻视野 |
| `resample_interval` | 3 | 10 | 每世界练 10 轮，配合 priority sampling |
| priority | uniform | `\|δp\| + \|δφ\|×5` | 偏离/碰撞场景优先采样 |
| offset path endpoint | 偏移 3.75m | 最后 5 步收敛到中心终点 | goal 检测触发 done |

**注意**：Issue 74 将此修复进一步完善为无条件最后 10% 步骤收敛（非依赖 `valid_len`），因为对 95%+ 的正常世界 `valid_len==91`，旧逻辑永不会触发。

**涉及文件**：`base.yaml`、`per_buffer.py`、`idc-train.py`、`idc_state_builder.py`

---

## 74. Transformer 滑窗架构替换 MLP Actor

**日期**：2026-06-02

**背景**：MLP 3 层 512→512→256 训练 500 轮 PDMS ~50，转向仍然差。根本问题是 MLP 逐帧盲推——每帧独立回归，没有时间记忆。转向是 2 阶控制问题，依赖累积转角，前馈网络没有积分器。

**方案选型**：评估了 LSTM+hidden snapshot (B+C) 和 Transformer 滑窗 (D)。选 D — Transformer 固定窗口 O(1)、训练/推理完全一致、无需 hidden state snapshot 管理。

**架构**：

```
TransformerActor(state_dim=62, d_model=256, nhead=4, num_layers=2, window=16)

输入: [batch, 16, 62]   # 最近 1.6 秒完整状态
  → Linear(62→256)
  → + learnable position encoding(16, 256)
  → TransformerEncoder(2 layers, 4 heads, FFN 1024, GELU, dropout 0.1)
  → 取最后一帧: [batch, 256]
  → Linear(256→2) → [steer_raw, acc_raw]
```

**关键设计决策**：

| 决策 | 理由 |
|------|------|
| window_size=16 | 1.6 秒上下文，3× 转向时间常数 |
| Critic 保持 MLP | V(s) 是 Markov 瞬时估值，无需时序 |
| f_pred_batch 不动 | 动力学推演仍是单帧接口 |
| rollout 滑窗 | `window_{t+1} = [window_t[1:] || f_pred_batch(window_t[-1], actor(window_t))]` |
| 窗口前端 pad | 重复第一帧（非零填充），避免 attention 被零向量干扰 |
| BC loss 仍用初始窗口 | BC 只在真实帧上算 steer supervision |

**PER Buffer 重构**：

| 改前 | 改后 |
|------|------|
| SumTree.data 存 Python object 元组 | 存 int 索引 |
| 经验 = (state[62], w, p) | 经验 = (window[16,62], w, p) |
| sample_batch 返回 `(states, worlds, paths)` | 返回 `(windows, worlds, paths)` |
| 新增 ~760 MB 窗口数组（200k × 16 × 62 × 4B） | 序列化同步 |

**窗口维护**：

- `agent.state_windows[world_idx] = deque(maxlen=16)` 每步 push 新状态
- `select_action` 从 deque 拼窗口 → batched forward
- `agent.reset_world_state(w)` 在 epoch 开始、world done/bad/reached、resample 时调用

**涉及文件**（7 个）：

| 文件 | 改动 |
|------|------|
| `continuous_actor_critic.py` | +84 行 TransformerActor（保留 ContinuousActor） |
| `per_buffer.py` | SumTree.data int 索引，+windows 数组，改 add/sample/序列化 |
| `idc_agent.py` | Actor→TransformerActor，select_action/update/rollout 全改窗口流转 |
| `idc-train.py` | 缓冲插入前拼窗口，epoch/resample 时 reset |
| `idc-eval.py` | epoch/filter 后 reset 窗口 |
| `pdms.py` | RolloutPDMSScorer 改为窗口 rollout |
| `base.yaml` | +window_size/d_model/nhead/num_layers/dropout 配置 |

---

## 75. Rollout 窗口退化修复：horizon 收窄 + rho 涨速控制

**日期**：2026-06-02

**症状**：Transformer 训练 100 轮：actor loss 振荡于 100↔5000（无收敛趋势），critic loss 早期 300k+，PDMS 5-37 宽幅震荡。每步 PIM 都触发 violation → rho 0.01 单调升至 1.76。LA-DIAG 显示 rollout 后期出现 18-35m 跟踪误差。

**根因**：30 步 rollout × window=16。步 16 之后窗口 100% 是预测帧 → Transformer self-attention 在腐败数据上做决策 → 误差几何放大。rho 涨速太快（+0.005/PIM）→ penalty 迅速支配 loss → actor 再也学不到 tracking。

**修复**：

| 参数 | 旧 | 新 | 效果 |
|------|-----|-----|------|
| `horizon` | 30 | 16 | rollout 在窗口全腐败前终止 |
| `amplifier_c` | 0.005 | 0.002 | rho 涨速降 60%，给 tracking 更多时间 |

horizon=16 步 × 0.1s = 1.6 秒，恰好等于窗口长度——第 16 步窗口才切换到全预测帧，但 rollout 已结束。

**涉及文件**：`base.yaml`

---

## 76. GEP rho 从累加器改为 EMA（经过 4 轮迭代）

**日期**：2026-06-02

**背景**：GEP 的 rho 调节经历了 4 轮迭代，每轮都在同一累加器模式下调参，从未收敛：

| 轮次 | 方案 | 失败原因 |
|:---:|------|------|
| 1 | 单向累加（ampl=0.005，无 cap） | rho→3.0 闷死 tracking |
| 2 | 双向累加 + scale cap=10 | 增减比 100:1，早期不掉头 |
| 3 | 双向累加 + warmup=300 | warmup 太长，无 penalty 信号 → 学到转圈 |
| 4 | 双向累加 + warmup=0 | epoch 64 rho=2.07，仍然失控 |

**根因**：累加器模型（`rho += ampl * scale`）本质上 ρ 只涨不记"何时降"，与模型是否在进步完全解耦。早期大量违规 → ρ 必然爆到 cap，再小的 ampl、再高的 cap 都只是在延后而非制止。

**最终方案：EMA（指数移动平均）替代累加器**

```python
# 违规时：EMA 快速响应
target = init_penalty + min(max_p, 50.0) * 0.02    # [0.01, 1.01]
rho = min(0.95 * rho + 0.05 * target, max_penalty)

# 干净时：EMA 缓慢遗忘
rho = max(0.99 * rho + 0.01 * init_penalty, init_penalty)
```

数学性质：
- rho 跟踪近期违规严重度的指数加权均值（非历史总和）
- 违规率稳定时 rho 收敛到稳态值，不漂移
- 违规率降至 0 时 rho 指数衰减回 init_penalty（半衰期约 70 PIM ≈ 23 epoch）

**连带删除**：
- `warmup_pims` — 不再需要（EMA 自带平滑，早期不爆炸）
- scale cap (`min(..., 10.0)`) — 不再需要（target 公式自带 soft cap）
- `amplifier_c` — 标记为已弃用

**涉及文件**：`idc_agent.py:update()`、`base.yaml`

