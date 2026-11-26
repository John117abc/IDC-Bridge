#!/bin/bash
# plan2 配置下跑一条路线，并生成"相机 + 规划面板"双联 gif
# 用法：eval_panel_plan2.sh <ckpt> <route_idx> <tag>
set -u
R=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge
PY=/mnt/2T_HDD/conda_envs/bench2drive_b2d/bin/python
D=/mnt/2T_HDD/plant2_vector

CK=${1:?ckpt}
IDX=${2:?idx}
TAG=${3:-plan2}

for p in $(pgrep -f "CarlaUE4|leaderboard_evaluator" 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
sleep 4
rm -rf $D/video/* 2>/dev/null; rm -f $D/dbg_*.npz

export VEC_CTRL=idc VEC_ROUTE_ONLY=0 VEC_REF_FROM=plan VEC_REF_MODE=plan2
export VEC_PLAN2_LEN=28 VEC_PLAN2_VFLOOR=3.0 VEC_IDC_LONG_REF=1
export VEC_CKPT="$CK"
export VEC_IDC_CKPT=/mnt/2T_HDD/WaymoDatabas/checkpoints/bc/idc_60ep.pth
export VEC_STUCK_SEC=25 VEC_DEBUG=1 VEC_RECORD=1 VEC_SAFETY=1
export VEC_IDC_VMAX=14 VEC_IDC_VMIN=0.0
export VEC_LAT_MODE=pp VEC_PP_MAX=0.75 VEC_PP_RATE=0.35
export ROUTES_FILE=$D/other_towns.xml
export ROUTE_IDX=$IDX
export LIMIT=200
export RUN_NAME=panel_${TAG}_$(date +%H%M%S)
bash $R/tools/fast_test.sh > $D/panel_${TAG}.log 2>&1

echo "=== gif 合成 ==="
DBG=$(ls $D/dbg_0.npz 2>/dev/null || ls $D/dbg_1.npz 2>/dev/null)
if [ -n "$DBG" ]; then
  $PY $R/tools/compose_panel_gif2.py --video $D/video --dbg "$DBG" \
      --out $D/panel_${TAG}.gif --every 2 --max 150 2>&1 | tail -3
  ls -la --time-style=+%H:%M $D/panel_${TAG}.gif 2>/dev/null
else
  echo "没有 dbg，无法合成"
fi
RD=$(ls -dt $D/closedloop/* | head -1)
echo "run: $RD"
grep -E "RouteCompletionTest|CollisionTest|MinSpeedTest" "$RD/run.log" 2>/dev/null | tail -3
