#!/bin/bash
# 固定评测流程：跑指定路线集 → 自动生成"相机 + 规划面板"双联 gif
# 用法：eval_with_panel.sh <ckpt> <tag> <routes.xml> <每路线超时秒>
set -u
R=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge
PY=/mnt/2T_HDD/conda_envs/bench2drive_b2d/bin/python
D=/mnt/2T_HDD/plant2_vector

CKPT=${1:?ckpt}
TAG=${2:?tag}
ROUTES=${3:-$D/other_towns.xml}
RTIMEOUT=${4:-600}

for p in $(pgrep -f "CarlaUE4-Linux-Shipping" 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
sleep 3
rm -rf $D/video/* 2>/dev/null; rm -f $D/dbg_*.npz

export VEC_CTRL=idc VEC_ROUTE_ONLY=0 VEC_REF_FROM=route
export VEC_CKPT="$CKPT"
export VEC_IDC_CKPT=/mnt/2T_HDD/WaymoDatabas/checkpoints/bc/idc_60ep.pth
export VEC_STUCK_SEC=25 VEC_DEBUG=1 VEC_RECORD=1 VEC_SAFETY=1
export VEC_IDC_VMAX=14 VEC_IDC_VMIN=5.5
export VEC_LAT_MODE=pp VEC_PP_MAX=0.75 VEC_PP_RATE=0.35
export ROUTES_FILE="$ROUTES"
export RUN_NAME=${TAG}_$(date +%H%M%S)
export RUN_TIMEOUT=$RTIMEOUT
bash $R/tools/run_b2d_smoke.sh

DBG=$(ls $D/dbg_0.npz 2>/dev/null || ls $D/dbg_1.npz 2>/dev/null)
if [ -n "$DBG" ]; then
  $PY $R/tools/compose_panel_gif2.py --video $D/video --dbg "$DBG" \
    --out $D/panel_${TAG}.gif --every 2 --max 150 2>&1 | tail -2
fi
