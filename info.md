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
