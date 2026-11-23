#!/bin/bash
set -u
ROOT=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge
CK=$(ls -t /mnt/2T_HDD/WaymoDatabas/checkpoints/20261122/*.pth 2>/dev/null | head -1)
CK=${CK:-/mnt/2T_HDD/WaymoDatabas/checkpoints/20261119/idc-waymo-v1.0_examples_041839_episode=475.pth}

export VEC_CTRL=idc
export VEC_ROUTE_ONLY=0          # ← 上层规划器参与
export VEC_IDC_CKPT="$CK"
RUN=${RUN_NAME:-twostage_$(date +%H%M%S)}; export RUN_NAME=$RUN
echo "权重=$CK  RUN=$RUN  (规划器 + IDC)"
bash $ROOT/tools/run_b2d_smoke.sh > /tmp/carla_twostage.log 2>&1
echo "exit=$?"
bash $ROOT/tools/show_results.sh /mnt/2T_HDD/plant2_vector/closedloop/$RUN 2>&1 | grep -E 'route Route|infractions' | head -4
echo "--- agent 侧关键日志 ---"
grep -E 'VecAgent|IDC\]|Traceback' /tmp/carla_twostage.log | head -8
