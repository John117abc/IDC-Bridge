# 横向多模态候选：设计与实施计划（v0.1）

> 分支：`research/lateral-modes`
> 状态：设计定稿，待实施
> 决策记录：2026-11-15

---

## 0. 三条已定决策

| # | 决策 | 说明 |
|---|---|---|
| 1 | 候选来源 = **采样 + 聚类** | 先在 Frenet 空间大量采样 → 过滤 → 聚类取 K 个中心。不训练生成器（阶段一） |
| 2 | 证书 **只做过滤**，不参与选哪条 | 选路由 IDC 的 `J`（模型前瞻代价）或 `V`（critic）负责 |
| 3 | 独立分支 `research/lateral-modes` | 验证通过后再合并主线；主线保持 `baseline-v1-2400ep` 不变 |

### 为什么证书不能参与选择

`ego_action_capability` 里"动作选择有效性 v0.1"的 verdict 是：

```
未进 Full：SOURCE_ACTION_LEAKAGE / ACTION_INVERSE_SHORTCUT / REGION_SUPPORT_DEGENERATE
```

让 capability 参与"选哪条"会泄漏源动作。**过滤**判的是与策略无关的属性（这条轨迹本身可行吗、和别的真的不同吗），没有这个问题。

---

## 1. 架构

```
专家路径 + 道路图 + 周车
        │
        ▼
 ┌──────────────────┐   ① 采样：横向偏移 × 速度缩放
 │ candidate_sampler │      → M 条原始候选（M ≈ 30~60）
 └──────────────────┘
        │
        ▼
 ┌──────────────────┐   ② 过滤：4 条判据 + 三态
 │ candidate_certifier│     → INFEASIBLE 丢弃；UNKNOWN 保守保留但标记
 └──────────────────┘
        │
        ▼
 ┌──────────────────┐   ③ 聚类：确定性 FPS，取 K 个中心
 │  (sampler 内)     │      → K = 3~5 条互不相同的候选
 └──────────────────┘
        │
        ▼  K 条 Candidate
 ┌──────────────────┐   ④ 选一条：J（阶段一）→ V（阶段二）
 │   mode_selector   │      + 滞后阈值 + ε-greedy 探索
 └──────────────────┘
        │  1 条参考
        ▼
 ┌──────────────────┐
 │ 现有 GEP 跟踪器    │  → [acc, steer]   ← 已封版，不改
 └──────────────────┘
```

职责边界：

| 层 | 判什么 | 与策略相关？ |
|---|---|---|
| 证书 | 可行 + 互不相同 | ❌ 无关 |
| `J` / `V` | 哪个代价更小 | ✅ 相关 |
| 跟踪器 | 怎么执行 | ✅ 相关 |

---

## 2. 模块与接口

新增（**不改主线模块**）：

```
src/env/candidate_sampler.py     # ① 采样  ③ 聚类
src/env/candidate_certifier.py   # ② 四判据 + 三态
src/env/mode_selector.py         # ④ J/V 选择 + 滞后 + ε-greedy
tools/check_candidates.py        # M0 验收：叠加可视化 + 统计
```

### 2.1 数据结构

```python
@dataclass
class Candidate:
    param: dict            # {"lat_offset": float, "v_shift": float, "cluster_id": int}
    pos: np.ndarray        # [T, 2]   世界坐标
    heading: np.ndarray    # [T]
    vel: np.ndarray        # [T, 2]   由曲率限速重采样
    road_dist: np.ndarray  # [T]
    status: str            # CERTIFIED_DIFFERENT / CERTIFIED_FEASIBLE / INFEASIBLE / UNKNOWN
    cert: dict             # 每条判据的明细（便于排查）
```

### 2.2 采样器（决策 1 的落地）

采样维度刻意保持**低维**，这样聚类有意义：

| 维度 | 取值 | 作用 |
|---|---|---|
| `lat_offset` | −3.0 … +3.0，步长 0.5（13 个） | 横向：避让 / 换道倾向 |
| `v_shift` | {0.7, 0.85, 1.0} × 曲率限速 | 纵向：让行 / 抢行 |

`M = 13 × 3 = 39` 条原始候选。生成方式：

```
对每个 (lat_offset, v_shift):
  pos[i] = expert_pos[i] + lat_offset · n[i]        # n 为逐点法向量
  走廊裁剪：若 road_dist(pos[i]) < d_safe，沿法向拉回
  重算 heading / 曲率限速 / road_dist
```

**注意**：`lat_offset` 是相对**专家路径**的偏移，不是相对自车。这样参考不会跟着自车跑（避免"参考贴着车走 → 误差恒为 0"的退化）。

### 2.3 聚类（确定性 FPS）

**不用 k-means**，用确定性最远点采样（FPS）——与 `ego_action_capability` 的"一切选择由确定性 FPS / 分层 / self-distance 决定"纪律一致，保证可复现。

```
距离度量：两条轨迹的 Hausdorff 距离（世界坐标，逐点）
初始化：取 road_dist 最大（最接近车道中心）的可行候选
迭代：每次加入"与已选集合距离最大"的候选，直到 K 条
```

### 2.4 证书：4 条判据 + 三态

| # | 判据 | 定义 | 默认阈值 |
|---|---|---|---|
| 1 | 运动学可行 | 逐点曲率 `\|κ\| ≤ κ_max` 且 `\|a\| ≤ a_max` | `κ_max` 用法从**数据实测**（专家轨迹曲率 95 分位），不拍脑袋 |
| 2 | 走廊约束 | `road_dist(t) ≥ d_safe` 逐点 | `d_safe = 0.8 m` |
| 3 | 碰撞 | 与回放周车最小距离 `≥ d_col`，或 `TTC ≥ 0.5 s` | `d_col = 1.0 m` |
| 4 | 差异性 | 与已选候选的 Hausdorff `≥ δ` | `δ = 1.5 m` |

三态（沿用 `mul_mod_traj` 的命名与纪律）：

```
INFEASIBLE          违反 1~3
CERTIFIED_FEASIBLE  可行，但与已有候选同模态
CERTIFIED_DIFFERENT 可行 且 与已有候选确实不同
UNKNOWN             判据不适用 / 数值不可靠 → 保守处理，不算作"不同"
```

**诚实指标**：报告 `K'`（真实不同的条数）而不是名义 `K`；`K'` 是下界。

### 2.5 选择器（决策 2 的落地）

```python
class ModeSelector:
    def __init__(self, k, interval=15, eps=0.2, delta=0.05):
        ...
    def score_by_J(self, ...):    # 阶段一：对 K 条各 rollout H 步，比较 Σγ^t(l+ρp)
    def score_by_V(self, ...):    # 阶段二：对 K 条各算 critic 值
    def select(self, scores, prev):   # 硬选 + 滞后：V(新) < V(当前) − δ·|V| 才切换
```

**阶段一用 `J`**：零训练、从第一步就能用、没有"V 没见过候选 → 永远选一条"的死锁。

---

## 3. 里程碑与验收标准

| 阶段 | 做什么 | 验收标准 | 预估 |
|---|---|---|---|
| **M0** | 采样 + 证书 + 聚类 + `check_candidates.py` | 20 个场景上 `K'` 稳定在 3~5；每条都过判据 1~3；**叠加图人眼确认可行驶**；统计"候选与专家的最大横向偏离"与"候选两两最大间距" | 2~3 天 |
| **M1** | `mode_selector`（J 版）+ 全量落盘 | 纯 eval（**不改训练**）能出"非零偏移选中率"；落盘 `(state, K 候选, 全部 J, 选中 index)` | 3~5 天 |
| **M2** | 训练接入（ε-greedy + 周期重选） | 400 epoch 对照：PDMS 不降、碰撞步数下降、切换频率 < 0.2 次/秒 | 1~2 周 |
| **M3** | 选择器换/加 `V` | 远视是否带来提升 | 3~5 天 |

**止损点**：M0 若出现 `K' ≤ 1`（聚类后候选全一样）或"走廊裁剪把候选全压回专家"，**停下来改采样空间**，不要进入 M1。

---

## 4. 与主线的关系

- 主线文件**不改**：`idc_agent.py` / `idc_state_builder.py` / `world_manager.py` 均不动
- 只在 train/eval 里加 **1~2 行钩子**（把 `generate_candidate_paths` 的结果替换为候选集）
- 分支可随时 `git checkout` 回 `baseline-v1-2400ep` 复现基线
- 合并条件：M2 的对照结果优于基线，且 `selftest_idc.py` 全绿

---

## 5. 风险与对策

| 风险 | 表现 | 对策 |
|---|---|---|
| 采样空间选错 | K 条候选只有速度差异，没有横向差异 | M0 统计"候选两两最大横向间距"；若 < 1 m 则扩大采样范围 |
| 走廊裁剪过度 | 所有候选被压回专家路径，`K' = 1` | M0 统计"裁剪后与专家的最大偏离"；若 < 0.5 m 则放宽 `d_safe` 或改采样 |
| 横向抖动 | 相邻步来回切候选 | 滞后阈值 `delta` + 最小重选间隔 |
| 证书太严 | 全部 UNKNOWN / INFEASIBLE | 先只启用判据 1~2，判据 3 用较宽松阈值起步 |
| 与既有评估不可比 | 换了场景采样 seed | 对照实验固定同一个 seed 与场景集 |

---

## 6. 待办（按顺序）

- [ ] M0-a：实测专家轨迹曲率分布，定 `κ_max`
- [ ] M0-b：`candidate_sampler.py`（采样 + 走廊裁剪 + 曲率限速重采样）
- [ ] M0-c：`candidate_certifier.py`（4 判据 + 三态）
- [ ] M0-d：确定性 FPS 聚类
- [ ] M0-e：`tools/check_candidates.py`（叠加图 + 统计表）
- [ ] M0-f：20 场景验收，出报告
