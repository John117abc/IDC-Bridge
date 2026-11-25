#!/bin/bash
# 夜间收尾：等训练完成 → 4 路线闭环评测（新 occ 规划器）→ 若通过则跑 24 路线泛化
set -u
PY=/mnt/2T_HDD/conda_envs/bench2drive_b2d/bin/python
T=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/tools
D=/mnt/2T_HDD/plant2_vector
LOG=$D/night_finalize.log
REP=$D/night_finalize_report.txt
CKPT=$D/ckpt_occ/best.pt

echo "[$(date +%H:%M:%S)] 收尾脚本启动，等待训练完成" >> "$LOG"

# 1) 等训练结束
while true; do
  if grep -q "训练完成" "$D/train_occ.log" 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] 检测到训练完成" >> "$LOG"; break
  fi
  if ! pgrep -f "train_plant2_occ.py" > /dev/null 2>&1 && grep -q "ep " "$D/train_occ.log" 2>/dev/null; then
    sleep 120
    if ! pgrep -f "train_plant2_occ.py" > /dev/null 2>&1; then
      echo "[$(date +%H:%M:%S)] 训练进程消失，视为结束" >> "$LOG"; break
    fi
  fi
  sleep 120
done

ls -la "$CKPT" >> "$LOG" 2>&1
grep -E "ep |早停|训练完成" "$D/train_occ.log" | tail -6 >> "$LOG"

# 2) 4 路线闭环
for p in $(pgrep -f "CarlaUE4-Linux-Shipping" 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
sleep 3
rm -rf $D/video/* 2>/dev/null; rm -f $D/dbg_*.npz
export VEC_CTRL=idc VEC_ROUTE_ONLY=0 VEC_REF_FROM=route
export VEC_CKPT="$CKPT"
export VEC_IDC_CKPT=/mnt/2T_HDD/WaymoDatabas/checkpoints/bc/idc_60ep.pth
export VEC_STUCK_SEC=25 VEC_DEBUG=1 VEC_RECORD=1 VEC_SAFETY=1
export VEC_IDC_VMAX=14 VEC_IDC_VMIN=5.5
export VEC_LAT_MODE=pp VEC_PP_MAX=0.75 VEC_PP_RATE=0.35
export ROUTES_FILE=$D/other_towns.xml
export RUN_NAME=occ4_$(date +%H%M%S)
export RUN_TIMEOUT=600
echo "[$(date +%H:%M:%S)] 开始 4 路线评测 RUN=$RUN_NAME" >> "$LOG"
bash $T/run_b2d_smoke.sh >> "$LOG" 2>&1
RD=$D/closedloop/$RUN_NAME
{
  echo "======== 新 occ 规划器 4 路线（$(date)）========"
  bash $T/show_results.sh $RD 2>&1 | grep -E 'route Route|infractions' | head -8
  echo "--- 指标 ---"
  $PY $T/analyze_dev.py 2>&1 | grep -E 'route 最近点|v 中位'
} >> "$REP" 2>&1
$PY $T/gif_from_frames.py $D/video $D/vis_occ4.gif 4 140 >> "$LOG" 2>&1
tail -12 "$REP" >> "$LOG"

# 3) 判断是否跑 24 路线泛化：4 条里至少 3 条完成
N_OK=$(grep -c "Completed" "$REP" | tail -1)
echo "[$(date +%H:%M:%S)] 4 路线完成数 $N_OK" >> "$LOG"
if [ "$N_OK" -ge 3 ]; then
  echo "[$(date +%H:%M:%S)] 进入 24 路线泛化测试" >> "$LOG"
  for p in $(pgrep -f "CarlaUE4-Linux-Shipping" 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
  sleep 3
  export ROUTES_FILE=$D/night_routes.xml
  export RUN_NAME=occ24_$(date +%H%M%S)
  export RUN_TIMEOUT=420
  bash $T/run_b2d_smoke.sh >> "$LOG" 2>&1
  RD2=$D/closedloop/$RUN_NAME
  {
    echo
    echo "======== 新 occ 规划器 24 路线（$(date)）========"
    bash $T/show_results.sh $RD2 2>&1 | grep -E 'route Route' | head -26
  } >> "$REP" 2>&1
fi
echo "[$(date +%H:%M:%S)] 夜间收尾全部完成" >> "$LOG"
