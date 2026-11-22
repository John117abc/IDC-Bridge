#!/bin/bash
echo "===== PlanT2 checkpoints 目录 ====="
find /mnt/2T_HDD/PlanT2.0/checkpoints -maxdepth 2 2>/dev/null | head -15
echo "===== 找 config.json ====="
find /mnt/2T_HDD/PlanT2.0 -maxdepth 4 -name 'config.json' 2>/dev/null | head -5
find /home/jiangchengxuan/PycharmProjects/research-safe-drive/plant2 -maxdepth 4 -name 'config.json' 2>/dev/null | head -5
echo "===== GlobalConfig 定义位置 ====="
grep -rIn --include=*.py -e "class GlobalConfig" /home/jiangchengxuan/PycharmProjects/research-safe-drive/plant2 2>/dev/null | head -3
echo "===== two_routes.xml 内容（前 15 行） ====="
head -15 /home/jiangchengxuan/PycharmProjects/research-safe-drive/plant2/Bench2Drive/leaderboard/data/two_routes.xml
echo "===== results_smoke/save 里有什么 ====="
ls /home/jiangchengxuan/PycharmProjects/research-safe-drive/plant2/Bench2Drive/results_smoke/save 2>/dev/null | head -5
