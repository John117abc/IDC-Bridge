#!/bin/bash
# 闭环评测：指定 ckpt + 路线，plan2 + LONG_REF
set -u
R=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge
D=/mnt/2T_HDD/plant2_vector
CK=${1:?ckpt}
IDX=${2:?route idx}
TAG=${3:-eval}
REP=$D/eval_${TAG}.txt

for p in $(pgrep -f "CarlaUE4|leaderboard_evaluator" 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
sleep 4
rm -rf $D/video/* 2>/dev/null; rm -f $D/dbg_*.npz
export VEC_CTRL=idc VEC_ROUTE_ONLY=0 VEC_REF_FROM=plan VEC_REF_MODE=plan2
export VEC_PLAN2_LEN=28 VEC_PLAN2_VFLOOR=3.0 VEC_IDC_LONG_REF=1
export VEC_CKPT=$CK
export VEC_IDC_CKPT=/mnt/2T_HDD/WaymoDatabas/checkpoints/bc/idc_60ep.pth
export VEC_STUCK_SEC=25 VEC_DEBUG=1 VEC_RECORD=1 VEC_SAFETY=1
export VEC_IDC_VMAX=14 VEC_IDC_VMIN=0.0
export VEC_LAT_MODE=pp VEC_PP_MAX=0.75 VEC_PP_RATE=0.35
export ROUTES_FILE=$D/other_towns.xml
export ROUTE_IDX=$IDX
export LIMIT=200
export RUN_NAME=ev_${TAG}_$(date +%H%M%S)
bash $R/tools/fast_test.sh > $D/eval_${TAG}.log 2>&1
RD=$(ls -dt $D/closedloop/fast_* | head -1)
{
  echo "==== $TAG : $(basename $(dirname $CK))/$(basename $CK) idx=$IDX ===="
  grep -E "RouteCompletionTest|OutsideRouteLanesTest|CollisionTest|MinSpeedTest" "$RD/run.log" | tail -4
  echo "-- 速度 --"
  grep -h "^\[PID\]" "$RD/run.log" | tail -2
} > $REP
cat $REP
