#!/bin/bash
# P0: dump online features at step 0 (4 worlds), then compare with offline build_one
set -u
cd /gpudrive/baselines/IDC-Bridge || exit 1
P=/gpudrive/.venv/bin/python
CKPT=/workspace/data/checkpoints/20261119/idc-waymo-v1.0_examples_041839_episode=475.pth
L=/workspace/data/logs
D=/data/data_json/training
PL=/workspace/data/ab60k_B/latest.pt

echo "### dump 在线特征（step 0）"
$P src/scripts/eval/idc-eval.py --data-dir $D --model-path "$CKPT" \
    --num-worlds 4 --seed 42 --planner-runtime "$PL" --planner-mode planner \
    --planner-replan-every 5 --planner-k 1 \
    --dump-feat /workspace/data/rt_feat.npz --gif-enabled false \
    > "$L/p0_dump.log" 2>&1
echo "exit=$?"
grep -E "FEAT" "$L/p0_dump.log" | tail -2

echo "### 对比"
$P tools/cmp_feat.py --dump /workspace/data/rt_feat.npz --n 4 2>&1 | tail -60
