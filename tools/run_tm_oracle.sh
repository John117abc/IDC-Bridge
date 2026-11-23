#!/bin/bash
set -u
ROOT=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge
PY=/mnt/2T_HDD/conda_envs/bench2drive_b2d/bin/python

echo "=== 语法检查 ==="
$PY - <<'EOF'
import ast
ast.parse(open('/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/tools/plant_agent_vec.py').read())
print('AST OK')
EOF
bash $ROOT/tools/sync_carla_modules.sh 2>&1 | tail -1

echo "=== TM oracle 跑 4 个城镇 ==="
export VEC_CTRL=tm
export VEC_ROUTE_ONLY=1
export VEC_DEBUG=1
export VEC_STUCK_SEC=25          # TM 可能遇红灯等待，放宽
export TM_SPEED=8.0
export ROUTES_FILE=/mnt/2T_HDD/plant2_vector/other_towns.xml
export RUN_NAME=tm_$(date +%H%M%S)
rm -f /mnt/2T_HDD/plant2_vector/dbg_*.npz
bash $ROOT/tools/run_b2d_smoke.sh > /tmp/tm_oracle.log 2>&1
echo "exit=$?"
bash $ROOT/tools/show_results.sh /mnt/2T_HDD/plant2_vector/closedloop/$RUN_NAME 2>&1 | grep -E 'route Route|infractions' | head -6
echo "=== TM 初始化 ==="
grep -h '\[TM\]' /mnt/2T_HDD/plant2_vector/closedloop/$RUN_NAME/*.log 2>/dev/null | head -4
echo "=== 横向误差 ==="
$PY $ROOT/tools/analyze_dev.py 2>&1 | grep -E 'route 最近点|v 中位'
