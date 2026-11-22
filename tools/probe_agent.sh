#!/bin/bash
R=/home/jiangchengxuan/PycharmProjects/research-safe-drive/plant2
echo "===== DataAgent 结构 ====="
grep -nE "^class |^    def " "$R/carla_garage/data_agent.py" | head -30
echo "===== PID / 控制器 ====="
grep -rIl --include=*.py -e "class PID" -e "class LateralPIDController" -e "class Controller" "$R" 2>/dev/null | grep -v third_party | head -6
echo "===== PlanT2 模型定义 ====="
ls "$R/PlanT" 2>/dev/null | head -12
grep -rIl --include=*.py -e "class PlanT" -e "class PlanT2" "$R" 2>/dev/null | grep -v third_party | head -6
echo "===== 评测入口（leaderboard 脚本） ====="
ls "$R/leaderboard/scripts" 2>/dev/null | head -10
ls "$R/Bench2Drive" 2>/dev/null | head -12
echo "===== 现成的 agent 实现（team_code） ====="
find "$R" -maxdepth 3 -type d -name 'team_code' 2>/dev/null | head -4
find "$R" -maxdepth 4 -name '*agent*.py' 2>/dev/null | grep -v third_party | head -10
echo "===== 已有评测结果目录（说明命令历史） ====="
ls "$R/Bench2Drive/results_full" 2>/dev/null | head -5
