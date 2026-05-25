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
| 世界重采样 | **坏世界 > `--max-bad-worlds` 时自动触发** | 从全量池随机抽取新世界 |
| 数据池大小 | **`--dataset-size` 自动检测 data_dir 下 tfrecord 文件数** | 0=自动, 手动指定可覆盖 |

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
