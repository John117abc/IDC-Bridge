#!/bin/bash
set -u
# CARLA 闭环：VEC_CTRL=idc，跑 2 条直路
ROOT=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge
CK84=$(ls -t /mnt/2T_HDD/WaymoDatabas/checkpoints/20261122/*.pth 2>/dev/null | head -1)
CK64=/mnt/2T_HDD/WaymoDatabas/checkpoints/20261119/idc-waymo-v1.0_examples_041839_episode=475.pth
CK=${VEC_IDC_CKPT:-${CK84:-$CK64}}
echo "IDC 权重 = $CK"
[ -f "$CK" ] || { echo "权重不存在"; exit 1; }

export VEC_CTRL=idc
export VEC_IDC_CKPT="$CK"
RUN=${RUN_NAME:-idc_$(date +%H%M%S)}; export RUN_NAME=$RUN
bash $ROOT/tools/run_b2d_smoke.sh > /tmp/carla_idc1.log 2>&1
echo "exit=$?"
bash $ROOT/tools/show_results.sh /mnt/2T_HDD/plant2_vector/closedloop/$RUN 2>&1 | grep -E 'route Route|infractions' | head -4
echo "--- agent 日志（IDC 初始化）---"
grep -E 'IDC\]|Traceback|Error' /tmp/carla_idc1.log | head -10
