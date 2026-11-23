#!/bin/bash
set -u
ROOT=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge
export VEC_CTRL=idc
export VEC_ROUTE_ONLY=1
export VEC_IDC_CKPT=/mnt/2T_HDD/WaymoDatabas/checkpoints/bc/idc_bc.pth
export VEC_RECORD=1
export VEC_DEBUG=1
export VEC_STUCK_SEC=0
export ROUTES_FILE=/mnt/2T_HDD/plant2_vector/other_towns.xml
export RUN_NAME=pid_vis_$(date +%H%M%S)
rm -rf /mnt/2T_HDD/plant2_vector/video/* 2>/dev/null
bash $ROOT/tools/run_b2d_smoke.sh > /tmp/pid_vis.log 2>&1
echo "exit=$?"
D=/mnt/2T_HDD/plant2_vector/closedloop/$RUN_NAME
bash $ROOT/tools/show_results.sh $D 2>&1 | grep -E 'route Route' | head -4
echo "=== PID 诊断（前 6 行 / 后 4 行）==="
grep -h '\[PID\]' $D/*.log 2>/dev/null | head -6
grep -h '\[PID\]' $D/*.log 2>/dev/null | tail -4
echo "=== 录像帧数 ==="
ls /mnt/2T_HDD/plant2_vector/video/ 2>/dev/null | wc -l
