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
| `amplifier_c` | 1.015~1.02 | ρ每次PIM放大倍率 |
| `lr_actor` | 1e-5 | |
| `lr_critic` | 3e-4 | |
| `horizon` | 25 | 推演步数 |
| `batch_size` | 256 | |
| `num_worlds` | 200 | |
