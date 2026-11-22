#!/bin/bash
R=/home/jiangchengxuan/PycharmProjects/research-safe-drive/plant2
F=$R/carla_garage/plant_agent.py
echo "===== plant_agent.py 结构 ====="
grep -nE "^class |^    def " "$F" | head -40
echo "===== 行数 ====="
wc -l "$F"
echo "===== 关键行：waypoint / pid / model ====="
grep -nE "waypoint|pid|PID|model\(|torch|route|global_plan|BEV|bev" "$F" | head -40
echo "===== plant.py（基类？）结构 ====="
F2=$R/carla_garage/plant.py
grep -nE "^class |^    def " "$F2" 2>/dev/null | head -30
echo "===== 已有评测命令线索 ====="
ls "$R/Bench2Drive/tools" 2>/dev/null | head -10
grep -rIn --include=*.sh -e "run_evaluation" -e "plant_agent" "$R" 2>/dev/null | grep -v third_party | head -8
