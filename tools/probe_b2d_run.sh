#!/bin/bash
R=/home/jiangchengxuan/PycharmProjects/research-safe-drive/plant2
echo "===== PlanT2 checkpoint 目录 ====="
ls -la /mnt/2T_HDD/PlanT2.0/checkpoints/PlanT2 2>/dev/null | head -10
echo "===== 路线文件（xml） ====="
find "$R/Bench2Drive/leaderboard/data" -name '*.xml' 2>/dev/null | head -10
find "$R/leaderboard/data" -name '*.xml' 2>/dev/null | head -6
echo "===== run_evaluation.sh 参数 ====="
sed -n '1,40p' "$R/Bench2Drive/leaderboard/scripts/run_evaluation.sh" 2>/dev/null
echo "===== 之前跑过的命令线索（shell history / 脚本） ====="
grep -rIn --include=*.sh -e "run_evaluation.sh" "$R/Bench2Drive/leaderboard/scripts" 2>/dev/null | head -5
ls "$R/Bench2Drive/results_smoke" 2>/dev/null | head -5
