# Waymo 真实数据外部检验方案
## ——验证 PlanT 中“低阶纵向规划响应结构”是否可能来自真实驾驶行为，而非仅是模型偏置

> **定位**  
> 本文不是用 Waymo 数据“证明 Driving Relation”，也不是把 CARLA 的 I1–I5 反事实干预硬搬到真实数据上。  
> 当前唯一目标是做一个 **Real-World External Reality Check**：
>
> **Town12 / PlanT 中发现的“跨场景低阶纵向 planning-response structure”，在真实道路驾驶日志中是否也能观察到类似现象？**

---

# 1. 当前已知结论与本实验要回答的问题

CARLA + PlanT 的 Gate 6.12A/B/C 已经把原始现象逐层拆开：

1. 规划响应几乎完全由纵向变化构成，横向贡献很小；
2. 未归一化纵向响应主要受 endpoint progress 和 intervention magnitude 影响；
3. 去掉幅值后剩余的 horizon shape 也基本可由低阶时间模板解释；
4. 当前最稳妥的表述是：

> **在 Town12 / PlanT 中观察到了跨场景重复的低阶纵向 planning-response envelope。**

现在最大的未决问题是：

> **这是实际驾驶任务本身具有的低复杂度结构，还是 PlanT 的 route-conditioned planner / waypoint decoder / 当前 intervention 设计共同造成的模型偏置？**

Waymo 实验只回答这一点。

---

# 2. 数据集选择

## 2.1 主数据集

使用：

**Waymo Open Motion Dataset（WOMD）v1.3.1，Scenario proto 格式。**

优先使用：

- `training`
- `validation`

暂时不用 `test`，因为 test 不提供未来真实轨迹。

WOMD 每个 Scenario 提供：

- 1 s 历史；
- 当前状态；
- 8 s 未来；
- 10 Hz；
- SDC（自车）轨迹；
- 其他交通参与者轨迹；
- lane / boundary / crosswalk / traffic signal 等地图信息；
- `sdc_track_index`；
- `objects_of_interest`；
- v1.3.1 中新增的 `sdc_paths`。

本实验**不需要相机、LiDAR，也不需要 GPU**。

---

## 2.2 一个必须注意的事实

WOMD 官方说明其数据是：

> manually-driven 与 autonomously-driven 的未标注混合数据。

因此本文中不要写：

- “人类专家轨迹”
- “human driver response”

统一写：

> **real-world logged SDC / ego behavior**

否则证据强度会被写过头。

---

# 3. 总体实验结构

整个 Waymo 分析只分四个 Gate：

```text
W0  数据读取与坐标系统验证
        ↓
W1  真实驾驶纵向/横向调整分析
        ↓
W2  低阶时间结构分析
        ↓
W3  Interaction vs Non-interaction 对照
        ↓
结论：更像真实驾驶结构 / 更像 PlanT 偏置 / 仍无法区分
```

不要在这一阶段：

- 训练预测模型；
- 学 Relation；
- 人工设计 TTC / clearance Relation；
- 做 counterfactual generation；
- 上 Waymax；
- 同时加入 nuScenes。

先把这个最小 Reality Check 做干净。

---

# 4. Gate W0：数据链路与坐标系统验证

## 4.1 目标

确认能够稳定提取：

```text
scenario_id
timestamps
SDC track
other tracks
object type
map
sdc_paths / lane geometry
traffic signals
objects_of_interest
```

并将 Ego 未来轨迹转换到**道路局部坐标系**。

---

## 4.2 为什么不能直接使用 global x / y

Waymo 的 global x/y 是 East/North。

如果车辆正在左转：

```text
global x/y 都会发生明显变化
```

这并不代表驾驶员在做“横向规避”。

因此必须把 Ego 轨迹投影到道路/路线参考坐标中：

\[
s(t)=\text{沿参考路径前进距离}
\]

\[
d(t)=\text{相对参考路径的横向偏移}
\]

后续：

- \(s\)：纵向；
- \(d\)：横向。

这样才能和当前 PlanT 的 longitudinal / lateral 分解对应。

---

# 5. Reference path 的确定

## 5.1 首选

如果使用 WOMD v1.3.1：

优先使用 `sdc_paths` 中标记为可用 / on-route 的 SDC route path。

选择规则必须在看结果前固定：

1. 只考虑有效 on-route path；
2. 用 SDC 当前点到 path 的最近距离；
3. 用当前 SDC heading 与 path 切线方向差；
4. 选择距离最小且方向一致的 path；
5. **不得使用未来 Ego 行为来挑“最像真实轨迹”的 path。**

避免 future leakage。

---

## 5.2 fallback

如果当前 Scenario 没有可靠 `sdc_paths`：

使用附近 lane centerline 构建 reference path。

如果仍无法稳定建立 reference：

```text
该 Scenario 直接排除
```

不要人工修。

---

# 6. 分析 horizon

主分析统一使用：

\[
T=5\,s
\]

即当前时刻之后 50 个 future samples。

原因：

- 5 s 足够覆盖典型交互响应；
- 比 8 s 更少受到路线分叉、长时行为变化影响；
- 与后续其他真实驾驶数据更容易统一。

8 s 只做 sensitivity check，不作为主结果。

---

# 7. Gate W1：真实驾驶是否也主要是纵向调整？

## 7.1 先定义“行为调整”，而不是直接看轨迹

不能直接使用：

\[
s(t),d(t)
\]

因为正常行驶本身就会前进。

需要建立一个**无新增调整的运动基线**。

---

## 7.2 Baseline A：Constant Velocity（CV）

根据当前 SDC longitudinal speed：

\[
s_{CV}(t)=s_0+v_{s,0}t
\]

横向：

\[
d_{CV}(t)=d_0
\]

真实轨迹残差：

\[
\Delta s_{CV}(t)
=
s_{real}(t)-s_{CV}(t)
\]

\[
\Delta d(t)
=
d_{real}(t)-d_0
\]

---

## 7.3 Baseline B：Constant Acceleration（CA）

这是**必须增加的对照**。

因为如果只用 CV，那么：

\[
\Delta s(t)\approx at^2
\]

可能只是车辆正常保持加速度的结果。

利用过去 1 s SDC 历史估计当前 longitudinal：

\[
v_0,\quad a_0
\]

构造：

\[
s_{CA}(t)
=
s_0+v_0t+\frac12 a_0t^2
\]

然后：

\[
\Delta s_{CA}(t)
=
s_{real}(t)-s_{CA}(t)
\]

主报告必须同时给：

```text
CV residual
CA residual
```

如果低阶结构只存在于 CV residual，而 CA residual 基本消失，则说明主要只是普通运动学，而不是新的驾驶响应结构。

---

# 8. Longitudinal Dominance 指标

对每一个有效 state：

\[
E_s=
\sum_t \Delta s(t)^2
\]

\[
E_d=
\sum_t \Delta d(t)^2
\]

定义：

\[
R_{long}
=
\frac{E_s}{E_s+E_d+\epsilon}
\]

分别计算：

```text
R_long_CV
R_long_CA
```

统计：

- median；
- Q25 / Q75；
- frac(\(R_{long}>0.9\))；
- 不同 interaction group；
- interaction vs control。

---

# 9. 不能直接对全 Waymo 做统计

因为：

> 大部分普通道路驾驶本来就是纵向行驶。

如果把所有 Scenario 混在一起，得到：

\[
R_{long}\approx1
\]

没有研究意义。

因此必须建立：

```text
Interaction Set
vs
Matched Non-Interaction Control
```

这是本方案最重要的设计之一。

---

# 10. Interaction Set 的构造

## 10.1 第一轮只使用结构化轨迹，不人工看视频

对每个 Scenario：

1. 读取 `objects_of_interest`；
2. 如果没有，允许从有效动态 actor 中补候选；
3. 保留当前距离 Ego 较近的 actor；
4. 检查 actor 未来 5 s 轨迹与 Ego reference path / Ego future corridor 是否存在明显空间接近；
5. 选择最相关的一个 actor 作为 `target_actor`。

注意：

> target_actor 选择规则只是 **sample screening**，不是 Relation 定义。

---

## 10.2 推荐的最小几何筛选

第一版先固定：

```text
当前 actor-Ego 距离 <= 50 m
actor 至少有 3 s 有效 future
SDC 至少有完整 5 s future
actor future 与 SDC route corridor 的最小距离 <= 5 m
```

如样本过少，只允许统一放宽一次：

```text
5 m → 7.5 m
```

并记录版本。

不要边看结果边调阈值。

---

# 11. Interaction mechanism 分组

这里只用于“跨物理机制统计”，不是 Relation。

建议建立四组，与当前 CARLA 研究尽量对应，但名称不要强行完全一致。

## G1：Lead / Longitudinal Interaction

典型：

- actor 为 vehicle；
- 位于 Ego 前方；
- 与 Ego path 大致同向；
- 存在减速、速度差或跟驰交互。

对应当前 HardBreak 的真实数据近似。

---

## G2：Merge / Cut-in

典型：

- vehicle 初始不在 Ego route corridor；
- 未来进入 Ego corridor；
- lateral offset 明显减小。

对应 ParkingCutIn。

---

## G3：Crossing / Turning Vehicle

典型：

- vehicle heading 与 Ego route 存在较大夹角；
- actor path 与 Ego route/corridor 有空间交叉或近交。

对应 VehicleTurning。

---

## G4：VRU Crossing

actor：

```text
pedestrian / cyclist
```

且：

- 轨迹靠近或穿过 Ego route corridor。

对应 PedestrianCrossing。

---

# 12. 不要一开始追求特别大的数据量

建议分两阶段。

## Pilot

先处理：

```text
2,000 个 Scenario
```

目标：

```text
每组至少 50 个有效 interaction state
总 interaction >= 200
```

如果达不到，再扩展到：

```text
5,000 → 10,000 Scenario
```

不要一开始扫完整 WOMD。

这个任务主要是 CPU + 磁盘 IO，不需要占 A4000。

---

# 13. Gate W2：真实纵向调整是不是也只有低阶时间结构？

对每个 interaction state 的：

\[
\Delta s_{CA}(t)
\]

优先做三种拟合。

## Model 0：Constant

\[
\Delta s(t)=c
\]

作为最低阶 baseline。

---

## Model 1：Linear

\[
\Delta s(t)=a t
\]

---

## Model 2：Quadratic

\[
\Delta s(t)=a t+b t^2
\]

---

# 14. 低阶模板解释率

定义：

\[
R_{energy}
=
1-
\frac{
\sum_t(\Delta s(t)-\hat{\Delta s}(t))^2
}{
\sum_t\Delta s(t)^2+\epsilon
}
\]

分别得到：

```text
E_const
E_linear
E_quadratic
```

统计：

```text
median
Q25/Q75
fraction > 0.8
fraction > 0.9
```

并分别统计四类 interaction group。

---

# 15. 这里真正要防止的错误解释

如果出现：

\[
R_{energy,quadratic}\approx1
\]

不能立即写：

> 真实驾驶也存在低阶 Driving Relation。

因为车辆本身就是平滑动力学系统。

因此必须进入 Gate W3。

---

# 16. Gate W3：Matched Non-Interaction Control

## 16.1 Control 的目的

回答：

> 这种“纵向 + 二次模板”是不是所有正常驾驶轨迹本来就这样？

如果是，则它只是：

```text
road-following / vehicle kinematics
```

而不是 interaction-specific structure。

---

# 17. Control Set 构造

从没有明显 target interaction 的 Scenario 中选 Ego state。

与每个 interaction state 按以下变量匹配：

```text
当前 Ego speed
reference-path curvature
道路类型 / lane geometry（可用时）
future valid length
```

推荐：

```text
1 个 interaction
匹配 1~3 个 control
```

不需要完全精确 propensity matching。

第一版使用 nearest-neighbor matching 即可。

---

# 18. Control 中计算完全相同的指标

必须运行同一套：

```text
R_long_CV
R_long_CA

E_const
E_linear
E_quadratic

|Δs_CA| energy
|Δd| energy
```

绝对不能为 control 单独换算法。

---

# 19. 最关键的四张结果表

## 表 1：Longitudinal dominance

| Group | n | R_long_CA median | frac>0.9 |
|---|---:|---:|---:|
| G1 Lead | | | |
| G2 Cut-in | | | |
| G3 Crossing Vehicle | | | |
| G4 VRU | | | |
| Control | | | |

---

## 表 2：Adjustment magnitude

| Group | \(\|\Delta s_{CA}\|\) median | \(\|\Delta d\|\) median |
|---|---:|---:|
| G1 | | |
| G2 | | |
| G3 | | |
| G4 | | |
| Control | | |

这一张用于确认：

> interaction 确实引起了比 control 更明显的行为调整。

否则后面的 shape 分析没有意义。

---

## 表 3：Low-order explainability

| Group | Linear median | Quadratic median | frac Quad>0.9 |
|---|---:|---:|---:|
| G1 | | | |
| G2 | | | |
| G3 | | | |
| G4 | | | |
| Control | | | |

---

## 表 4：CARLA / Waymo 对照

| 现象 | PlanT Town12 | Waymo |
|---|---|---|
| longitudinal dominance | 很强 | 待测 |
| lateral contribution | 很低 | 待测 |
| low-order temporal fit | 很高 | 待测 |
| interaction-specific adjustment | counterfactual 可直接测 | 用 observational residual + control 间接测 |
| cross-scene repetition | 已观察 | 待测 |

---

# 20. 必须画的图

不要输出大量图，固定以下 6 类。

## Figure 1

随机选 20 个 interaction：

```text
reference path
SDC real future
target actor future
```

用于人工确认场景筛选合理。

---

## Figure 2

四类 interaction 的：

\[
R_{long,CA}
\]

分布图。

---

## Figure 3

Interaction vs Control：

\[
\|\Delta s_{CA}\|
\]

分布。

---

## Figure 4

Interaction vs Control：

\[
R_{energy,quadratic}
\]

分布。

---

## Figure 5

每组各选 5 个：

```text
真实 Δs_CA(t)
linear fit
quadratic fit
```

---

## Figure 6

PlanT 与 Waymo 的 summary comparison。

---

# 21. 必须增加的 QC

## QC1：Reference path projection

检查：

- 当前 SDC 到 reference path 距离；
- heading difference；
- 未来 5 s 是否出现异常 path jump。

异常直接排除。

---

## QC2：低响应状态

如果：

\[
E_s+E_d
\]

非常小，那么 \(R_{long}\) 没有解释意义。

因此把：

```text
almost-no-adjustment state
```

单独报告，不参加 direction-ratio 主统计。

阈值用 pilot 数据预先确定，例如使用 residual energy 的低 5% 分位，而不是看最终结果后手工选择。

---

## QC3：Turning geometry

至少人工检查：

```text
左转
右转
直行
```

各 10 个 Scenario。

确认 Frenet / route coordinate 没有把正常转弯误认为 lateral response。

---

## QC4：Future validity

5 s 内缺失过多：

```text
直接丢弃
```

不要插值长缺口。

---

# 22. 结果判定：只有三种情况

## Case A：Waymo 与 PlanT 基本一致

如果四类真实 interaction 中都观察到：

```text
明显 longitudinal dominance
+
interaction residual 明显高于 control
+
residual 仍可由低阶 temporal template 较好解释
```

则可以写：

> **PlanT 中观察到的低阶纵向 response structure 并非明显违背真实驾驶日志。真实不同交互机制中同样存在以纵向调整为主、且时间复杂度较低的行为结构。**

注意仍然不能写：

> “Driving Relation 已被真实数据证明。”

这只是 external consistency evidence。

---

## Case B：Waymo 明显比 PlanT 更“二维 / 高阶”

例如：

```text
大量 interaction 的 R_long 明显较低
明显 lateral adjustment
quadratic template 解释率明显下降
```

则应写：

> **Town12 中的低阶纵向结构很可能受到 PlanT planner / route conditioning / intervention design 的显著限制。**

此时不应优先继续 Town13。

---

## Case C：Interaction 与 Control 几乎一样

例如：

```text
R_long 都很高
quadratic 都很好
但 interaction residual magnitude 并不比 control 大
```

那么最重要的结论是：

> **“纵向 + 低阶时间模板”可能只是普通 road-following / 车辆运动学性质，不能作为 Driving Relation 的证据。**

这是必须接受的负结果。

---

# 23. 为什么这一步暂时不做 Cross-Scene Pair Cosine

Waymo 没有 CARLA 那样的：

\[
X
\rightarrow
I_1...I_5
\rightarrow
\Delta W_1...\Delta W_5
\]

真实 counterfactual。

因此不能把：

```text
两个 Waymo 场景的真实轨迹相似
```

直接解释成：

```text
它们具有相同 Driving Relation
```

第一轮只验证：

> **低阶纵向 response phenomenon 是否在真实数据中出现。**

如果这一步得到正面结果，再进入第二阶段：

```text
Natural Variation / Matched Pairs
```

利用真实数据中相近但 actor 状态略有差异的场景，近似研究：

\[
\frac{\partial W_{ego}}{\partial X_{actor}}
\]

这才更接近 CARLA intervention response。

当前先不做。

---

# 24. 推荐工程目录

```text
waymo_reality_check/
├── config/
│   └── gate_w.yaml
├── scripts/
│   ├── 00_smoke_read.py
│   ├── 01_extract_scenarios.py
│   ├── 02_build_reference_path.py
│   ├── 03_select_interactions.py
│   ├── 04_build_controls.py
│   ├── 05_compute_residuals.py
│   ├── 06_fit_low_order.py
│   ├── 07_qc.py
│   └── 08_report.py
├── cache/
│   ├── scenario_index.parquet
│   ├── interaction_states.parquet
│   └── control_states.parquet
├── results/
│   ├── metrics_state.parquet
│   ├── metrics_group.csv
│   └── gate_w_summary.md
└── figures/
```

---

# 25. 推荐每个 state 保存的字段

```text
scenario_id
timestamp/current_time_index
group
target_actor_id
target_actor_type

ego_speed
ego_acc
path_curvature

valid_future_sec
projection_error
heading_error

delta_s_cv[50]
delta_s_ca[50]
delta_d[50]

E_s_cv
E_s_ca
E_d

R_long_cv
R_long_ca

fit_const_score
fit_linear_score
fit_quadratic_score

fit_a
fit_b

is_interaction
matched_control_id
qc_pass
```

这样后续不需要反复重新扫 TFRecord。

---

# 26. 最小运行顺序

## Step 1

先读取 20 个 Scenario。

输出：

```text
SDC trajectory
map
sdc_paths
actors
objects_of_interest
```

人工确认数据理解正确。

---

## Step 2

处理 200 个 Scenario。

只验证：

```text
reference path
s/d 投影
CV / CA residual
```

如果这一步不稳定，禁止继续。

---

## Step 3

跑 2,000 Scenario Pilot。

目标：

```text
>=200 interaction states
每组尽量 >=50
```

生成全部 QC 图。

---

## Step 4

冻结：

```text
interaction screening
reference-path selection
horizon
control matching
metric
```

---

## Step 5

必要时扩展到：

```text
5,000~10,000 Scenario
```

不再修改规则。

---

## Step 6

输出 Gate W 结论。

只有在 Gate W 完成后，再决定：

```text
回 Town13
或
进入 Waymo Natural Variation 第二阶段
```

---

# 27. 第一阶段结束后最希望回答的一句话

不是：

> “Waymo 证明了 Driving Relation。”

而是：

> **PlanT 中发现的“跨场景低阶纵向 planning-response structure”，究竟与真实道路中的交互驾驶行为具有一致性，还是主要是 PlanT / 当前实验范式产生的结构？**

这就是 Waymo 第一阶段实验的唯一目标。

---

# 28. 当前建议的决策顺序

```text
Town12 Gate 6.12A/B/C
        ↓
已经发现：低阶纵向 planning-response envelope
        ↓
Waymo Gate W：Real-World Reality Check
        ↓
┌──────────────────────────────┐
│ 与真实交互行为一致？          │
└──────────────────────────────┘
       ↓ Yes                 ↓ No
Town13 replication       回查 PlanT bias
       ↓
Independent planner / Expert validation
       ↓
再讨论最终 Relation 定义
```

---

# 29. 暂时不要做的事情

在 Waymo Gate W 之前不要：

- 用 Waymo 训练新 planner；
- 直接把真实 Ego trajectory 当作 Relation；
- 直接用场景类别作为 Relation label；
- 做 PCA 后就声称“低维”；
- 只看 trajectory endpoint；
- 把二次轨迹拟合好解释成“关系存在”；
- 同时加入 nuScenes；
- 调参直到 Waymo 与 PlanT 得到一致结论。

---

# 30. 最终判定原则

这项 Waymo 分析必须允许出现负结果。

最有价值的不是“必须验证 PlanT”，而是明确区分：

\[
\text{真实驾驶任务结构}
\]

与

\[
\text{PlanT / route-conditioned planner 偏置}
\]

只有完成这一步，后续 Town13、独立 planner validation、Relation learning 才有清晰的科学意义。
