#!/usr/bin/env bash
# IDC 训练实时面板：每 30 秒刷新（Ctrl-C 退出）
# 用法： bash tools/watch_idc.sh          # 默认 30 秒
#        bash tools/watch_idc.sh 10       # 10 秒
INTERVAL=${1:-30}
LOGDIR=/mnt/2T_HDD/WaymoDatabas/logs

while true; do
  clear
  echo "================ IDC 训练面板  $(date '+%F %T')  ================"
  for run in g1_mlp g2_tr g3_bc; do
    L="$LOGDIR/$run.log"
    [ -f "$L" ] || continue
    echo ""
    echo "---------- $run ----------"
    grep -h "回合"      "$L" | tail -1 | sed 's/.*idc-train - //'
    grep -h "PHASE"     "$L" | tail -1 | sed 's/.*idc-train - //'
    grep -h "Critic Loss" "$L" | tail -1 | sed 's/.*idc-train - //'
    grep -h "DIAG-ref"  "$L" | tail -1 | sed 's/.*idc-train - //'
    grep -h "PDMS"      "$L" | tail -1 | sed 's/.*idc-train - //'
    # 最近 20 条 critic loss 的趋势（大→小 = 好）
    echo -n "critic 最近趋势: "
    grep -h "Critic Loss" "$L" | tail -20 | \
      sed 's/.*Critic Loss: *//; s/,.*//' | tr '\n' ' ' | cut -c1-120
    echo ""
  done
  echo ""
  echo "================================================================"
  echo "(每 ${INTERVAL}s 刷新；Ctrl-C 退出)"
  sleep "$INTERVAL"
done
