#!/bin/bash
set -u
B=/mnt/2T_HDD/PlanT2.0/PlanT2_Dataset/data/PlanT2_DS/PlanT_2_dataset/data
T=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/tools
M=/mnt/2T_HDD/plant2_vector/maps
V=/mnt/2T_HDD/plant2_vector/route_views
PY=/mnt/2T_HDD/conda_envs/bench2drive_b2d/bin/python
mkdir -p "$V"
i=0
for SC in HighwayCutIn SignalizedJunctionRightTurn NonSignalizedJunctionRightTurn PedestrianCrossing HighwayCutIn HardBreakRoute; do
  R=$(find "$B/$SC" -maxdepth 1 -mindepth 1 -type d -name 'Town1[23]*' 2>/dev/null | sed -n "$((i/2+1))p")
  [ -z "$R" ] && R=$(find "$B/$SC" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | head -1)
  [ -z "$R" ] && continue
  TOWN=$(basename "$R" | cut -d_ -f1)
  MP="$M/map_${TOWN}.npz"
  [ -f "$MP" ] || { echo "no map $TOWN"; i=$((i+1)); continue; }
  echo "=== $SC / $TOWN"
  "$PY" "$T/build_route_carla.py" --route-dir "$R" --map "$MP" --out "$V/r$i.png" 2>&1 | tail -2
  i=$((i+1))
done
ls "$V" | head
