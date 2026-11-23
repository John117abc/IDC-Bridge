#!/bin/bash
set -u
R=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge
PY=/mnt/2T_HDD/conda_envs/bench2drive_b2d/bin/python
echo "=== 语法检查 ==="
$PY - <<'EOF'
import ast
for f in ('plant_agent_idc', 'plant_agent_vec', 'idc_carla', 'frames', 'idc_state_carla'):
    p = f'/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/tools/{f}.py'
    ast.parse(open(p).read())
    print('OK', f)
EOF
echo "=== 同步到 carla_garage ==="
bash $R/tools/sync_carla_modules.sh 2>&1 | tail -2
echo "=== 跑 CARLA 闭环（IDC）==="
bash $R/tools/run_carla_idc.sh 2>&1 | tail -10
