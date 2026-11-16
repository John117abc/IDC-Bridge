#!/usr/bin/env bash
# 逐个配置起独立进程（每个进程一个 env），避免跨 rollout 污染
export MADRONA_MWGPU_KERNEL_CACHE=/gpudrive/gpudrive_cache
P=/gpudrive/.venv/bin/python
S=/gpudrive/baselines/IDC-Bridge/tools/diag_speed_geometry.py
CKPT=/workspace/data/checkpoints/20261115/idc-waymo-v1.0_examples_201537_episode=3000.pth

run() {
    geo=$1; spd=$2
    echo "=== running geo=$geo speed=$spd $(date '+%T') ==="
    "$P" "$S" --data-dir /data/data_json/training --model-path "$CKPT" \
        --num-worlds 32 --seed 0 --geo "$geo" --speed "$spd" \
        > "/workspace/data/logs/dg_${geo}_${spd}.log" 2>&1
    echo "exit=$? $(date '+%T')"
}

run expert ref
run map ref
run map expert
run expert const
echo "ALL DONE"
