#!/bin/bash
set -u
B=/mnt/2T_HDD/PlanT2.0/PlanT2_Dataset/data/PlanT2_DS/PlanT_2_dataset/data
T=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/tools
O=/mnt/2T_HDD/plant2_vector/vector_ds
V=/mnt/2T_HDD/plant2_vector/views
PY=/mnt/2T_HDD/conda_envs/bench2drive_b2d/bin/python
mkdir -p "$O" "$V"

i=0
for SC in HighwayCutIn SignalizedJunctionRightTurn NonSignalizedJunctionRightTurn PedestrianCrossing; do
  R=$(find "$B/$SC" -maxdepth 1 -mindepth 1 -type d -name 'Town1[23]*' 2>/dev/null | head -1)
  [ -z "$R" ] && { echo "skip $SC"; continue; }
  TOWN=$(basename "$R" | cut -d_ -f1)
  MAP=/mnt/2T_HDD/plant2_vector/maps/map_${TOWN}.npz
  [ -f "$MAP" ] || { echo "no map $TOWN"; continue; }
  OUT="$O/multi_${SC}.npz"
  echo "=== $SC ($TOWN) ==="
  "$PY" "$T/build_vector_dataset.py" --route "$R" --map "$MAP" --out "$OUT" 2>&1 | grep -E '样本数|周车|gt 前3|失败'
  [ -f "$OUT" ] || continue
  for IDX in 40 100; do
    "$PY" "$T/plot_vector_sample.py" --npz "$OUT" --idx $IDX \
      --out "$V/${SC}_$IDX.png" 2>&1 | tail -1
  done
  i=$((i+1))
done
ls -la "$V" | head -12
