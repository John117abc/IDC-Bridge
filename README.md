# IDC-Bridge

把"集成决策与控制"（IDC, Integrated Decision and Control）论文的算法跑在 Waymo 开放数据集 + GPUDrive 模拟器上。

我们复现了下游的在线强化学习流程——不照搬原论文的 30 个异步 learner 那套，而是用一个简单的单线程训练循环，跑 200 条道路场景同时仿真。

## 效果

训练过程会自动画轨迹对比图（专家轨迹 vs 实际跑出来的 vs 贝塞尔候选路径），方便盯着看训练完到底学会了没。

## 这个东西在干啥

IDC 这个算法的核心思想是**一边开一边学**。自车有两条线在看：

- **PEV（策略评估）**：用 Critic 网络评估当前状态有多好，每步都更新。
- **PIM（策略改进）**：每隔 N 步用 Actor 网络推演未来 H 步的轨迹，算出这条轨迹的成本（跟踪误差 + 碰撞风险），然后反向传播到 Actor 里改进策略。每改进一次就调大一次碰撞惩罚系数 ρ。

Actor 和 Critic 是两个独立的小 MLP 网络，动作空间是加速度和方向盘转角。

## 和原论文不一样的地方

先说和原论文的做法差不多是什么意思，再说哪里变了：

1. **网络结构**：原论文是前面一个共享的 latent embed，后面接 Actor 和 Critic 两个头。我们图省事，Actor 和 Critic 各搞各的，没共享参数。
2. **训练架构**：原论文用了 30 个异步 learner 从同一个重放缓冲区抽数据。我们一个主循环一边推环境一边训练，简单粗暴。
3. **环境接口**：原来是 Nocturne 模拟器，我们换成了 GPUDrive（Waymo 数据驱动）。
4. **候选参考路径**：原论文是构造一条专家参考轨迹。我们改成了用 Cubic Bezier 曲线从专家轨迹的起止点生成 3 条候选路径（左/中/右），训练时随机选一条做参考路径。
5. **碰撞模型**：原文用一个圆近似整车，我们改成前后双圆模型（车身前半段和后半段各一个圆），碰撞计算更精确。
6. **速度坐标系**：统一用车体坐标系的速度 v_lon/v_lat，比直接用全局 vx/vy 更符合运动学模型。
7. **性能优化**：原始的逐世界循环改成批量张量操作，状态构建每步只拉 3 次 GPU 数据而非 800 次（200世界×4次查询）。
8. **周车观察**：原始论文在每个场景观察的车作了分类，本代码直接取周围最近的8辆车。

## 文件说明

```
.
├── src/
│   ├── agents/idc_agent.py          # 智能体主逻辑：选动作、更新网络、GEP循环
│   ├── env/idc_state_builder.py     # 状态构建：从 GPUDrive 观测拼 IDC 格式，贝塞尔路径生成
│   ├── models/
│   │   ├── kinematic_bicycle.py     # 运动学自行车模型（状态推演用）
│   │   └── continuous_actor_critic.py # Actor/Critic 网络
│   ├── buffer/per_buffer.py         # 优先级经验重放缓冲区
│   ├── scripts/train/
│   │   ├── idc-train.py             # 训练入口
│   │   └── idc-eval.py              # 评估入口（确定策略，不训练）
│   └── utils/
│       ├── traj_visualizer.py       # 轨迹对比图
│       ├── visualr_recorder.py      # GIF 录制
│       └── ...
├── config.yaml                      # 还没用上的配置文件
└── requirements.txt                 # 依赖
```

## 快速开始

### 安装

```bash
pip install gpudrive torch numpy matplotlib
```

### 训练

```bash
python src/scripts/train/idc-train.py \
  --data-dir /path/to/waymo/data \
  --device cuda \
  --num-worlds 200 \
  --batch-size 256 \
  --horizon 25 \
  --epochs 100 \
  --pim-interval 30 \
  --init-penalty 1.0 \
  --amplifier-c 1.02 \
  --file-dir /workspace/data
```

### 评估

```bash
python src/scripts/train/idc-eval.py \
  --data-dir /path/to/waymo/data \
  --model-path /workspace/data/checkpoints/.../idc-waymo-v1.0_xxx.pth \
  --device cuda \
  --num-worlds 200
```

评估时会用确定策略（不加噪声），选中心的那条贝塞尔路径做参考。跑完后在 `traj_plots/` 下生成轨迹对比图。

### 关键参数

| 参数 | 含义 | 建议值 |
|------|------|--------|
| `--horizon` | 在线推演步数 | 25 |
| `--pim-interval` | 多少次 PEV 后做一次 PIM | 30 |
| `--init-penalty` | 初始碰撞惩罚系数 ρ | 1.0 |
| `--amplifier-c` | 每次 PIM 后 ρ 的放大倍率 | 1.015 ~ 1.02 |
| `--max-penalty` | ρ 上限 | 100 |
| `--lr-actor` | Actor 学习率 | 1e-5 |
| `--lr-critic` | Critic 学习率 | 3e-4 |

## 踩过的坑

1. **动作维度写反**：环境要 `[加速度, 转向角, 0]`，之前输出顺序是反的。这意味着模型一边推演一边跑环境看到的信号都是错的。
2. **sqrt 梯度 NaN**：当两车双圆心恰好重叠（距离 = 0），`torch.sqrt(0)` 的梯度是 `1/0 = inf`，链式法则里分子也是 0，`inf × 0 = NaN`。这个 NaN 顺着 `clip_grad_norm_` 污染了网络全部参数。解决办法是给所有 `torch.sqrt` 加一个 `1e-8` 的小 epsilon。
3. **车身系速度**：原来直接用 Waymo 原始数据的全局 vx/vy 算速度误差，没考虑到航向角和参考系的问题。改成 `[v_lon, v_lat]` 车身坐标系后一切正常。

## 引用

如果用了这个项目，请引用：

- 原版 IDC 论文：[Integrated Decision and Control: Toward  Interpretable and Computationally  Efficient Driving Intelligence](https://arxiv.org/abs/2103.10290)
- GPUDrive：[GPUDrive: Data-driven, multi-agent driving simulation at 1 million FPS](https://arxiv.org/abs/2408.01584)
