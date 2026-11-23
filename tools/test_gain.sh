#!/bin/bash
set -u
ROOT=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge
PY=/mnt/2T_HDD/conda_envs/bench2drive_b2d/bin/python
CK=$(ls -t /mnt/2T_HDD/WaymoDatabas/checkpoints/20261123/*.pth 2>/dev/null | head -1)

try_gain() {
  local g=$1
  export VEC_CTRL=idc VEC_ROUTE_ONLY=1 VEC_IDC_CKPT="$CK"
  export VEC_DEBUG=1 VEC_STUCK_SEC=6 VEC_IDC_STEER_GAIN=$g
  export RUN_NAME=gain_${g}_$(date +%H%M%S)
  rm -f /mnt/2T_HDD/plant2_vector/dbg_*.npz
  bash $ROOT/tools/run_b2d_smoke.sh > /tmp/gain_$g.log 2>&1
  echo "--- steer_gain=$g ---"
  bash $ROOT/tools/show_results.sh /mnt/2T_HDD/plant2_vector/closedloop/$RUN_NAME 2>&1 | grep -E 'route Route' | head -2
  $PY $ROOT/tools/analyze_dev.py 2>&1 | grep -E 'route 最近点|x=10m|plan 末点|v 中位|throttle'
}

echo "=== 基线 gain=1.0 ==="
try_gain 1.0
echo "=== 补偿 gain=1.6 ==="
try_gain 1.6
