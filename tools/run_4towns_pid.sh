#!/bin/bash
# 4 城镇验证（PID 执行层），总硬超时 420 s，卡死 6 s 即终止单条路线
set -u
ROOT=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge
export VEC_CTRL=idc
export VEC_ROUTE_ONLY=${VEC_ROUTE_ONLY:-1}
export VEC_IDC_CKPT=/mnt/2T_HDD/WaymoDatabas/checkpoints/bc/idc_bc.pth
export VEC_STUCK_SEC=${VEC_STUCK_SEC:-6}
export VEC_DEBUG=1
export VEC_IDC_VMAX=${VEC_IDC_VMAX:-12}
export ROUTES_FILE=/mnt/2T_HDD/plant2_vector/other_towns.xml
export RUN_NAME=pid4_$(date +%H%M%S)
echo "RUN=$RUN_NAME  (超时 420s, 卡死 6s, vmax=$VEC_IDC_VMAX)"
rm -f /mnt/2T_HDD/plant2_vector/dbg_*.npz
timeout 420 bash $ROOT/tools/run_b2d_smoke.sh > /tmp/pid4.log 2>&1
echo "exit=$? (124=整体超时)"
D=/mnt/2T_HDD/plant2_vector/closedloop/$RUN_NAME
bash $ROOT/tools/show_results.sh $D 2>&1 | grep -E 'route Route|infractions' | head -6
echo "=== 横向误差 / 车速 ==="
/mnt/2T_HDD/conda_envs/bench2drive_b2d/bin/python $ROOT/tools/analyze_dev.py 2>&1 | grep -E 'route 最近点|v 中位|throttle'
