#!/usr/bin/env bash
# A/B：同一批场景、同一组固定候选，只改"全部世界走哪条"
set -u
export MADRONA_MWGPU_KERNEL_CACHE=/gpudrive/gpudrive_cache
PY=/gpudrive/.venv/bin/python
EVAL=/gpudrive/baselines/IDC-Bridge/src/scripts/eval/idc-eval.py
MODEL=/workspace/data/checkpoints/20261115/idc-waymo-v1.0_examples_201537_episode=3000.pth
OUT=/workspace/data

for idx in 1 0 2; do
    tag=$([ "$idx" = "1" ] && echo center || { [ "$idx" = "0" ] && echo left08 || echo right08; })
    "$PY" "$EVAL" --data-dir /data/data_json/training --model-path "$MODEL" \
        --file-dir "$OUT" --num-worlds 32 --epochs 1 --seed 0 \
        --gif-enabled false --cand-fixed 3 --force-path-index "$idx" \
        > "$OUT/logs/ab_${tag}.log" 2>&1
    echo "[AB] idx=$idx tag=$tag exit=$?"
done
echo "[AB] done"
