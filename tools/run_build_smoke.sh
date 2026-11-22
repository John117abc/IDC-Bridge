#!/bin/bash
set -u
B=/mnt/2T_HDD/PlanT2.0/PlanT2_Dataset/data/PlanT2_DS/PlanT_2_dataset/data
T=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/tools
O=/mnt/2T_HDD/plant2_vector/vector_ds
mkdir -p "$O"

# 找一条 Town13 的 route（直行优先：HighwayCutIn / HardBreakRoute）
for SC in HighwayCutIn HardBreakRoute SignalizedJunctionRightTurn; do
  R=$(find "$B/$SC" -maxdepth 1 -mindepth 1 -type d -name 'Town1[23]*' 2>/dev/null | head -1)
  [ -n "$R" ] && { echo "scenario=$SC route=$R"; break; }
done
[ -z "${R:-}" ] && { echo "没找到 route 目录"; exit 1; }
TOWN=$(basename "$R" | cut -d_ -f1)
echo "town=$TOWN"
MAP=/mnt/2T_HDD/plant2_vector/maps/map_${TOWN}.npz
ls -la "$MAP" || exit 1

PY=""
for p in /mnt/2T_HDD/conda_envs/*/bin/python; do
  [ -x "$p" ] && "$p" -c "import numpy" >/dev/null 2>&1 && { PY="$p"; break; }
done
[ -z "$PY" ] && PY=/usr/bin/python3
echo "python=$PY"

"$PY" "$T/build_vector_dataset.py" --route "$R" --map "$MAP" --out "$O/smoke_$(basename $TOWN).npz" 2>&1 | tail -20
