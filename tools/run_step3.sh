#!/bin/bash
# 步骤3 验证：2D 动作空间（off_acc）4 城镇
set -u
ROOT=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge
export VEC_CTRL=idc
export VEC_ROUTE_ONLY=0
export VEC_REF_FROM=route
export VEC_IDC_CKPT=/mnt/2T_HDD/WaymoDatabas/checkpoints/bc/idc_2d.pth
export VEC_STUCK_SEC=6
export VEC_DEBUG=1
export VEC_IDC_VMAX=12
export ROUTES_FILE=/mnt/2T_HDD/plant2_vector/other_towns.xml
export RUN_NAME=step3_$(date +%H%M%S)
echo "RUN=$RUN_NAME ckpt=$(basename $VEC_IDC_CKPT)"
rm -f /mnt/2T_HDD/plant2_vector/dbg_*.npz
timeout 420 bash $ROOT/tools/run_b2d_smoke.sh > /tmp/step3.log 2>&1
echo "exit=$?"
D=/mnt/2T_HDD/plant2_vector/closedloop/$RUN_NAME
bash $ROOT/tools/show_results.sh $D 2>&1 | grep -E 'route Route|infractions' | head -6
echo "=== 动作空间确认 ==="
grep -h 'action_space' $D/*.log 2>/dev/null | head -2
echo "=== 横向/车速 ==="
/mnt/2T_HDD/conda_envs/bench2drive_b2d/bin/python $ROOT/tools/analyze_dev.py 2>&1 | grep -E 'route 最近点|v 中位'
