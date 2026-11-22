#!/bin/bash
set -u
A=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge
B=/home/jiangchengxuan/PycharmProjects/research-safe-drive/plant2
GN="jiangchengxuan"
GE="jiangchengxuan@users.noreply.github.com"

echo "========== 1) IDC-Bridge =========="
cd "$A" || exit 1
git add tools/*.py tools/*.sh \
        src/env/planner_features.py src/env/planner_runtime.py \
        src/scripts/eval/idc-eval.py src/utils/storyboard.py 2>/dev/null
git status --short | head -20
git -c user.name="$GN" -c user.email="$GE" commit -q -m "feat(plant2): 向量规划器接入 Bench2Drive 闭环

- 数据集: build_plant2_dataset / build_vector_dataset / build_route_carla
  (ego 23维 / agent 32x28 / map 128x7 / route 24x4 / gt 8x2)
- 规划器: train_planner 增加置信度头 + 锚点残差; 训练脚本 train_plant2_{small,full,conf}
- 闭环: plant_agent_vec.py (CARLA 0.9.15 + leaderboard, 2Hz 重规划, KD-tree 地图索引,
  曲率限速 + PI 速度控制 + 卡死检测, 三次样条密化)
- 评估: eval_plant2 / eval_modes; 可视化: plot_closedloop / plot_dbg / plot_route
- 运行时: planner_features (离线/在线同源特征), planner_runtime, storyboard 清理"
git log --oneline -1
echo "--- push ---"
git push 2>&1 | tail -4

echo
echo "========== 2) plant2 =========="
cd "$B" || exit 1
git add carla_garage/plant_agent_vec.py \
        Bench2Drive/leaderboard/leaderboard/autoagents/autonomous_agent_local.py 2>/dev/null
git status --short | head -10
git -c user.name="$GN" -c user.email="$GE" commit -q -m "feat: 闭环 agent (向量规划器 + 纯追踪)

- carla_garage/plant_agent_vec.py: 从 leaderboard AutonomousAgent 实现,
  输入 ego/周车/地图(npz)/route(XML 稠密路点), 输出 8 点 -> 样条密化 -> 控制
- 补 leaderboard 缺的 autonomous_agent_local 模块"
git log --oneline -1
echo "--- push ---"
git push 2>&1 | tail -4
