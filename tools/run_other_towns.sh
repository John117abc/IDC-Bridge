#!/bin/bash
set -u
ROOT=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge
CK=$(ls -t /mnt/2T_HDD/WaymoDatabas/checkpoints/20261123/*.pth 2>/dev/null | head -1)
export VEC_CTRL=idc
export VEC_ROUTE_ONLY=1
export VEC_IDC_CKPT="$CK"
export VEC_STUCK_SEC=${VEC_STUCK_SEC:-8}
export VEC_IDC_STEER_GAIN=${VEC_IDC_STEER_GAIN:-1.0}
export VEC_DEBUG=1
export ROUTES_FILE=/mnt/2T_HDD/plant2_vector/other_towns.xml
export RUN_NAME=towns_$(date +%H%M%S)
echo "权重=$CK  ROUTES=$ROUTES_FILE  RUN=$RUN_NAME"
rm -f /mnt/2T_HDD/plant2_vector/dbg_*.npz
bash $ROOT/tools/run_b2d_smoke.sh > /tmp/carla_towns.log 2>&1
echo "exit=$?"
bash $ROOT/tools/show_results.sh /mnt/2T_HDD/plant2_vector/closedloop/$RUN_NAME 2>&1 | grep -E 'route Route|infractions' | head -6
echo "=== 卡死记录 ==="
grep -h 'STUCK 车速' /mnt/2T_HDD/plant2_vector/closedloop/$RUN_NAME/*.log 2>/dev/null | head -4
