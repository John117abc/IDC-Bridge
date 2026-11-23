#!/bin/bash
set -u
ROOT=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge
PY=/mnt/2T_HDD/conda_envs/bench2drive_b2d/bin/python
export VEC_CTRL=idc
export VEC_ROUTE_ONLY=1
export VEC_IDC_CKPT=/mnt/2T_HDD/WaymoDatabas/checkpoints/bc/idc_bc.pth
export VEC_STUCK_SEC=10
export VEC_DEBUG=1
export ROUTES_FILE=/mnt/2T_HDD/plant2_vector/other_towns.xml
export RUN_NAME=bc_eval_$(date +%H%M%S)
echo "权重=$VEC_IDC_CKPT"
rm -f /mnt/2T_HDD/plant2_vector/dbg_*.npz
bash $ROOT/tools/run_b2d_smoke.sh > /tmp/bc_eval.log 2>&1
echo "exit=$?"
bash $ROOT/tools/show_results.sh /mnt/2T_HDD/plant2_vector/closedloop/$RUN_NAME 2>&1 | grep -E 'route Route|infractions' | head -6
echo "=== 横向偏移 / 车速 ==="
$PY $ROOT/tools/analyze_dev.py 2>&1 | grep -E 'route 最近点|v 中位|throttle'
