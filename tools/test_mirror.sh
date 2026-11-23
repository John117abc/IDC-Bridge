#!/bin/bash
set -u
ROOT=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge
PY=/mnt/2T_HDD/conda_envs/bench2drive_b2d/bin/python
CK=$(ls -t /mnt/2T_HDD/WaymoDatabas/checkpoints/20261123/*.pth 2>/dev/null | head -1)

try_mode() {
  local tag=$1; local nm=$2
  export VEC_CTRL=idc VEC_ROUTE_ONLY=1 VEC_IDC_CKPT="$CK"
  export VEC_DEBUG=1 VEC_STUCK_SEC=6 VEC_IDC_STEER_GAIN=1.0
  export VEC_IDC_NOMIRROR=$nm
  export RUN_NAME=${tag}_$(date +%H%M%S)
  rm -f /mnt/2T_HDD/plant2_vector/dbg_*.npz
  bash $ROOT/tools/run_b2d_smoke.sh > /tmp/$tag.log 2>&1
  echo "--- $tag (NOMIRROR=$nm) ---"
  bash $ROOT/tools/show_results.sh /mnt/2T_HDD/plant2_vector/closedloop/$RUN_NAME 2>&1 | grep -E 'route Route' | head -2
  $PY $ROOT/tools/analyze_dev.py 2>&1 | grep -E 'route 最近点|x=10m|v 中位|throttle'
}

echo "=== A) 正常镜像（baseline）==="
try_mode mir 0
echo "=== B) 关闭镜像（NOMIRROR=1）==="
try_mode nomir 1
