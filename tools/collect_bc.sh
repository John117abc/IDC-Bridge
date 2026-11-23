#!/bin/bash
set -u
ROOT=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge
PY=/mnt/2T_HDD/conda_envs/bench2drive_b2d/bin/python

echo "=== 语法检查 ==="
$PY - <<'EOF'
import ast
for f in ('plant_agent_vec', 'plant_agent_idc', 'idc_carla'):
    p = f'/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/tools/{f}.py'
    ast.parse(open(p).read()); print('OK', f)
EOF
echo "=== 同步 ==="
bash $ROOT/tools/sync_carla_modules.sh 2>&1 | tail -1

echo "=== 采集 BC 数据（纯追踪老师，4 条不同城镇路线）==="
rm -rf /mnt/2T_HDD/plant2_vector/bc; mkdir -p /mnt/2T_HDD/plant2_vector/bc
export VEC_CTRL=pp
export VEC_ROUTE_ONLY=1
export VEC_RECORD_BC=1
export VEC_BC_DIR=/mnt/2T_HDD/plant2_vector/bc
export VEC_STUCK_SEC=10
export VEC_DEBUG=0
export ROUTES_FILE=/mnt/2T_HDD/plant2_vector/other_towns.xml
export RUN_NAME=bc_$(date +%H%M%S)
bash $ROOT/tools/run_b2d_smoke.sh > /tmp/bc_collect.log 2>&1
echo "exit=$?"
bash $ROOT/tools/show_results.sh /mnt/2T_HDD/plant2_vector/closedloop/$RUN_NAME 2>&1 | grep -E 'route Route' | head -5
echo "=== 采集结果 ==="
ls -la /mnt/2T_HDD/plant2_vector/bc/ 2>/dev/null | head -6
grep -h '\[BC\]' /mnt/2T_HDD/plant2_vector/closedloop/$RUN_NAME/*.log 2>/dev/null | head -5
