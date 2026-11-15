#!/usr/bin/env bash
# Q1（相对车道中心的横向偏移分布）+ P1（运动学降级后重跑候选验收）
set -u
export MADRONA_MWGPU_KERNEL_CACHE=/gpudrive/gpudrive_cache
PY=/gpudrive/.venv/bin/python
R=/gpudrive/baselines/IDC-Bridge

"$PY" "$R/tools/check_lateral_offset.py" \
    --data-dir /data/data_json/training --num-worlds 40 --seed 0 \
    --out /workspace/data/plots/lateral > /workspace/data/logs/q1.log 2>&1
echo "[q1] exit=$?"

"$PY" "$R/tools/check_candidates.py" \
    --data-dir /data/data_json/training --num-worlds 24 --seed 0 \
    --plot-worlds 4 --out /workspace/data/plots/candidates2 \
    > /workspace/data/logs/p1.log 2>&1
echo "[p1] exit=$?"
echo "[done]"
