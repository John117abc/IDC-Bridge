#!/bin/bash
# Unified-interface comparison: route (4s slice of expert path) vs planner (4s model plan), world 1 and 6
set -u
cd /gpudrive/baselines/IDC-Bridge || exit 1
for W in ${WORLDS:-1 6}; do
  for M in route planner; do
    echo "===== $M / world $W ====="
    MODE=$M IDS=$W bash tools/run_multi_gif.sh
  done
done
