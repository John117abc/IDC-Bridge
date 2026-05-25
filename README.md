# IDC-Bridge

把"集成决策与控制"（IDC, Integrated Decision and Control）论文的算法跑在 Waymo 开放数据集 + GPUDrive 模拟器上。

使用单线程训练循环跑 200 条道路场景同时仿真的在线强化学习。

## 算法简介

IDC 的核心思想是**一边开一边学**：

- **PEV（策略评估）**：Critic 网络评估当前 state 的价值，每步更新。
- **PIM（策略改进）**：每隔 N 步，Actor 通过可微运动学模型推演未来 H 步轨迹，计算跟踪误差 + 碰撞风险的总成本，反向传播改进策略。每次改进放大惩罚系数 ρ。

Actor/Critic 为两个独立 MLP（LayerNorm + ELU），动作空间为加速度和方向盘转角。

## 和原论文的差异

1. **网络结构**：Actor/Critic 独立网络，无共享参数（原论文共享 latent embed）
2. **训练架构**：单线程主循环替代 30 个异步 learner
3. **环境**：Nocturne → GPUDrive（Waymo 数据）
4. **候选参考路径**：3 条 Cubic Bezier 路径（左/中/右 lateral offset），每 episode 固定选一条。道路宽度从 GPUDrive 动态读取
5. **碰撞模型**：前后双圆模型替代单圆
6. **状态空间**：59 维，含当前误差 + 3 个前瞻点（t+3/t+6/t+9）的横向/航向/道路边界预览
7. **性能优化**：批量 tensor 操作 + GPU `ref_tensor` 预索引，消除 rollout 中的 Python 循环瓶颈
8. **统一框架**：`rho=0` 即纯追踪诊断模式，`rho>0` 即含 collision/road penalty，无需额外 flag

## 快速开始

### 安装

```bash
pip install gpudrive torch numpy matplotlib
```

### 训练

```bash
# 纯追踪诊断（rho=0，无碰撞/道路惩罚）
python src/scripts/train/idc-train.py \
    --data-dir /path/to/waymo/data \
    --device cuda \
    --num-worlds 200 \
    --init-penalty 0 \
    --epochs 300 \
    --file-dir /workspace/data

# 正常训练（含 penalty）
python src/scripts/train/idc-train.py \
    --data-dir /path/to/waymo/data \
    --device cuda \
    --num-worlds 200 \
    --init-penalty 1.0 \
    --amplifier-c 1.015 \
    --max-penalty 10.0 \
    --epochs 300 \
    --file-dir /workspace/data
```

### 评估

```bash
python src/scripts/train/idc-eval.py \
    --data-dir /path/to/waymo/data \
    --model-path /workspace/data/checkpoints/.../idc-waymo-v1.0_xxx.pth \
    --device cuda \
    --num-worlds 10
```

### 关键 CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num-worlds` | 200 | 并行仿真世界数（固定不换） |
| `--horizon` | 20 | Rollout 推演步数 |
| `--dt` | 0.1 | 动力学时间步长（秒） |
| `--batch-size` | 512 | PEV/PIM 采样数 |
| `--hidden-dim` | 256 | Actor/Critic 隐藏层维度 |
| `--lr-actor` | 8e-5 | Actor 学习率 |
| `--lr-critic` | 3e-4 | Critic 学习率 |
| `--init-penalty` | 0 | 初始 ρ（0=纯追踪，1.0=含 penalty） |
| `--max-penalty` | 10 | ρ 上限 |
| `--amplifier-c` | 1.015 | ρ 每次 PIM 放大倍率 |
| `--pim-interval` | 30 | PEV 步数后执行一次 PIM |
| `--epochs` | 300 | 训练轮数 |
| `--save-freq` | 10 | 每 N epoch 保存一次 checkpoint |
| `--device` | cuda | 计算设备 |

### 诊断/调试 flags

| 参数 | 说明 |
|------|------|
| `--fix-speed` | 速度误差归零（诊断用） |
| `--fix-heading` | 航向误差归零（诊断用） |
| `--no-sign` | delta_p 去符号（诊断用） |

## State 布局（59 维）

| 区间 | 维度 | 内容 |
|------|------|------|
| [0:6] | 6 | 自车: x, y, v_lon, v_lat, phi, omega |
| [6:38] | 32 | 周车: 8 车 × (x, y, phi, v) |
| [38:46] | 8 | validity mask |
| [46:58] | 12 | ref_error: dp/dphi/dv + 3×(lat+dphi+road_dist) @ t+3/6/9 |
| [58:59] | 1 | temporal index |

## 文件结构

```
.
├── src/
│   ├── agents/idc_agent.py          # 智能体：动作选择、滚动推演、GEP 循环
│   ├── env/idc_state_builder.py     # 状态构建、Bezier 路径生成、参考点索引
│   ├── models/
│   │   ├── kinematic_bicycle.py     # 运动学自行车模型（L=5.0m，匹配 Waymo 实车）
│   │   └── continuous_actor_critic.py # Actor (LN+ELU×2+Tanh) / Critic (LN+ELU×2+Linear)
│   ├── buffer/per_buffer.py         # 经验回放缓冲区
│   ├── scripts/train/
│   │   └── idc-train.py             # 训练入口
│   ├── scripts/eval/
│   │   └── idc-eval.py              # 评估入口（确定性策略）
│   └── utils/
│       ├── traj_visualizer.py       # 轨迹对比图
│       ├── visualr_recorder.py      # GIF 录制
│       └── action_mapper.py         # 离散动作映射（暂未使用）
├── info.md                          # 完整开发记录（33 个 issue）
├── config.yaml
└── requirements.txt
```

## 踩过的坑

完整记录见 `info.md`。关键坑：

1. **动作维度写反**：环境要 `[acc, steer, 0]`，rollout 中通道颠倒 → 车持续画圆
2. **f_pred_batch 参考点 off-by-one**：动力学推进一步后参考点查询仍用旧索引 → heading error 180° 反向，400+ epoch 不收敛
3. **delta_phi 是 bearing error 不是 heading error**：`atan2(dy,dx)-ego` 应改为 `ref_h - ego_h`
4. **Wheelbase 硬编码 4.0m vs 真实车长 5.0m**：模型高估转弯能力 25%
5. **sqrt 梯度 NaN**：距离=0 时 `1/sqrt(0)=inf` → NaN 级联污染网络
6. **幽灵车**：GPUDrive 填零车被当真实车 → 虚假 penalty
7. **每步随机换候选路径**：参考点侧向跳动 7.5m → 追踪信号污染
8. **tracking_only 噪声不衰减**：σ=0.1 永不降 → 策略永远模糊

## 引用

- 原版 IDC 论文：[Integrated Decision and Control: Toward Interpretable and Computationally Efficient Driving Intelligence](https://arxiv.org/abs/2103.10290)
- GPUDrive：[GPUDrive: Data-driven, multi-agent driving simulation at 1 million FPS](https://arxiv.org/abs/2408.01584)
