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
self.net[6].bias.data[1] = 0.2  # tanh(0.2)≈0.197 → a≈0.3 m/s² 怠速蠕行
```
默认轻推，不够强到撞墙—碰撞 penalty 梯度远超偏置。

**注意**：必须删旧 checkpoint 重训，新初始化才生效。

---

## 7. 贝塞尔路径恒定速度

**现象**：红绿灯路口专家停了，但参考速度恒定（全局均值），Agent 不知道要停。

**根因**：`_make_bezier_path` 里 `speeds = np.full(num_points, speed)` — 所有 91 步同一个速度。

**修复**：
- `generate_candidate_paths` 改为传专家的时变速度 `expert_vel.norm(dim=-1).cpu().numpy()`
- `_make_bezier_path` 改为接收数组，直接使用：`speeds = np.asarray(expert_speeds)`

红绿灯路口专家速度降到 0 → 参考速度 = 0 → speed_err 推动 Agent 减速停车。

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

### 优化 3 — 诊断日志限频

`[DIAG-pos]` 从每次 penalty_batch 调用都打 → 每 50 步打一次。

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

## 关键参数速查

| 参数 | 值 | 说明 |
|------|-----|------|
| `D_veh_safe` | 2.0 | 双圆心碰撞阈值 |
| `D_road_safe` | 1.0 | 道路边界阈值 |
| `pim_interval` | 30 | 多少步PEV后做一次PIM |
| `amplifier_c` | 1.015 | ρ每次PIM放大倍率 |
| `lr_actor` | 3e-5 | |
| `lr_critic` | 3e-4 | |
| `horizon` | 25 | 推演步数 |
| `batch_size` | 256 | |
| `num_worlds` | 200 | |
| `noise_std` | 0.2 → 0.05 | 探索噪声，PIM 后 ×0.95 衰减 |
| `pos_err_weight` | 0.2 | 位置误差权重 |
| `heading_err_weight` | 0.3 | 朝向误差权重 |

---

## 10. Critic value 爆炸到 4300+

**现象**：训练中途 Critic value max 从 115 跳到 4296，loss 从 4000 跳到 19.8 万，之后持续震荡在 2400-4330，无法恢复。

**根因**：Critic 只有裸 `Linear+ELU`，无 LayerNorm。当异常世界状态（ego 坐标 15575m）进入时，随机权重 × 大输入 = 爆炸输出。MSE loss 巨大梯度破坏 Critic 权重，级联污染后续所有世界。

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

**现象**：`[DIAG-ref] max pos_err=15575m` 反复出现在不同 world。`[LARGE-ERR]` 显示 `ego=(-11000,-11000)`。

**诊断过程**：加了 5 种分层诊断标签：
- `[PATH-RANGE]`：训练开始时打印路径起点/终点坐标，确认坐标系
- `[LARGE-ERR]`：`get_idc_observations_batch` 中 pos_err > 100m 时打印 ego/ref
- `[FPRED-CLAMP]`：推演中位移 > 1m（后被取代）
- `[FPRED-ERR]`：推演中 delta_p > 100m
- `[TELEPORT]`：环境步后 abs(x\|y) > 1e5

**结论**：GPUDrive 使用本地坐标系（非 UTM），所有正常坐标在 ±100 范围内。约 2 个世界（world_40、world_105）的 ego 固定为 `(-11000, -11000)`，是特定数据的异常标记值，不是推演漂移产生。占比 1%，影响可忽略。

---

## 12. f_pred_batch clamp 策略演进

| 版本 | 锚点 | 行为 | 问题 |
|------|------|------|------|
| v1 | 相对上一步 `prev_x ± 10m` | 每步允许 10m 位移 | 25 步累积可达 250m，无法拦截跨步污染 |
| v2 | 相对参考路径 `ref_x ± 50m` | 以当前步参考路径为锚点 | 异常从第一步就被截断 |

关键设计：
- ref 查表必须在 clamp 之前（需要 ref 作为锚点坐标）
- clamp 后用 clamped 坐标计算 delta_p
- 窗口外（\|raw - ref\| > 50m）grad=0，不污染 Actor 梯度
- 50m 窗口：25 步正常推演最大位移 < 37m，50m 预留裕量

```python
# v2 核心逻辑
refs = get_ref_states_batch(w_i, x_raw.detach(), y_raw.detach(), ...)
ref_x, ref_y = refs[:, 0], refs[:, 1]
x_next = torch.clamp(x_raw, ref_x - 50, ref_x + 50)
y_next = torch.clamp(y_raw, ref_y - 50, ref_y + 50)
```

---

## 13. 无效世界过滤（已回滚）

**误判**：尝试用 `abs(path_start) < 5000` 过滤无效世界，但 GPUDrive 所有世界都是本地坐标（±100 内），导致 200/200 世界全被过滤。

**回滚**：删除 bad_worlds 检测和过滤逻辑，恢复全部世界参与训练。仅保留 `[LARGE-ERR]` 诊断。

**教训**：不了解坐标系时不要做硬过滤。GPUDrive 使用本地（relative）坐标，不是 UTM 绝对坐标。

---

## 14. Actor 转向策略锁定在右转

**现象**：训练 100+ 轮后 Critic loss 已收敛（5-20），但 20 个世界的 `norm_steer` 几乎全为负（右转），车在直道上也持续右偏。

**根因**：探索噪声 `std=0.05` 太小（转向噪声仅 `0.02 rad`），Actor 永远采样不到左转样本 → 梯度只反馈"右转还行" → 策略陷入局部最优。

**修复**：
| 改动 | 当前 | 改为 |
|------|------|------|
| 噪声 std | `0.05` 硬编码 | `self.noise_std = 0.2`，PIM 后衰减 `×0.95`，下限 `0.05` |
| 噪声变量 | 无 | `self.noise_std` / `self.noise_decay_rate` / `self.noise_std_min` |
| Actor 学习率 | `1e-5` | `3e-5` |

```python
# select_action
noise = torch.normal(0, self.noise_std, ...)

# update_actor 末尾
self.noise_std = max(self.noise_std_min, self.noise_std * self.noise_decay_rate)
```

---

## 15. 日志降噪

| 内容 | 原级别 | 现级别 | 频率 |
|------|--------|--------|------|
| `[DEBUG] target/value/loss stats` | INFO | DEBUG | 每步 |
| `回合 X/Y, 步数 Z/91` | INFO | INFO(10步)/DEBUG | 每步 |
| `[FPRED-CLAMP]` | 无 | WARNING | 异常时 |
| `[FPRED-ERR]` | 无 | WARNING | 异常时 |
| `[LARGE-ERR]` | 无 | WARNING | 异常时 |
| `[PATH-RANGE]` | 无 | INFO | 训练开始 |
| `[DIAG-act/ref/init/critic/pen]` | INFO | INFO | 保持
