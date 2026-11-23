#!/bin/bash
set -u
ROOT=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge
CK=$(ls -t /mnt/2T_HDD/WaymoDatabas/checkpoints/20261123/*.pth 2>/dev/null | head -1)
CK=${CK:-$(ls -t /mnt/2T_HDD/WaymoDatabas/checkpoints/20261122/*.pth | head -1)}

export VEC_CTRL=${VEC_CTRL:-idc}
export VEC_ROUTE_ONLY=${VEC_ROUTE_ONLY:-1}
export VEC_IDC_CKPT="$CK"
export VEC_DEBUG=1
export VEC_RECORD=1                 # 录车头画面
export VEC_STUCK_SEC=${VEC_STUCK_SEC:-6}
export RUN_NAME=stuck_$(date +%H%M%S)
echo "RUN=$RUN_NAME ctrl=$VEC_CTRL route_only=$VEC_ROUTE_ONLY"
rm -rf /mnt/2T_HDD/plant2_vector/video/* 2>/dev/null
bash $ROOT/tools/run_b2d_smoke.sh > /tmp/carla_stuck.log 2>&1
echo "exit=$?"
D=/mnt/2T_HDD/plant2_vector/closedloop/$RUN_NAME
echo "=== 结果 ==="
bash $ROOT/tools/show_results.sh $D 2>&1 | grep -E 'route Route|infractions' | head -4
echo "=== STUCK 诊断 ==="
grep -h 'STUCK-DIAG' $D/*.log 2>/dev/null | head -4
echo "=== 录像帧 ==="
ls -la /mnt/2T_HDD/plant2_vector/video/ 2>/dev/null | tail -5
