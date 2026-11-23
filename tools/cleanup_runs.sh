#!/bin/bash
set -u
echo "=== 残留的评测/服务进程 ==="
pgrep -af "leaderboard_evaluator|run_b2d_smoke|CarlaUE4|idc-train.py" 2>/dev/null | head -10
echo "=== 结束残留 agent 评测 ==="
pkill -f leaderboard_evaluator 2>/dev/null && echo killed evaluator || echo no evaluator
pkill -f run_b2d_smoke 2>/dev/null && echo killed smoke || echo no smoke
sleep 2
echo "=== 剩余 ==="
pgrep -af "leaderboard_evaluator|CarlaUE4" 2>/dev/null | head -5
echo "=== CARLA 服务是否还在 ==="
pgrep -af CarlaUE4 2>/dev/null | head -2
