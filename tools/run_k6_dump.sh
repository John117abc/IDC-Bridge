#!/bin/bash
# K=6 滚动重规划 + dump，用于可视化"多模态 + 选择 + 重规划"
set -u
cd /gpudrive/baselines/IDC-Bridge || exit 1
P=/gpudrive/.venv/bin/python
CKPT=/workspace/data/checkpoints/20261119/idc-waymo-v1.0_examples_041839_episode=475.pth
L=/workspace/data/logs
D=/data/data_json/training
PL=/workspace/data/ab60k_B/latest.pt

echo "### K=6, force index 0, dump"
$P src/scripts/eval/idc-eval.py --data-dir $D --model-path "$CKPT" \
    --num-worlds 4 --seed 42 --planner-runtime "$PL" --planner-mode planner \
    --planner-replan-every 5 --planner-k 6 --force-path-index 0 \
    --planner-dump /workspace/data/rt_dump.npz --gif-enabled false \
    > "$L/rt_k6.log" 2>&1
echo "exit=$?"
grep -E "Driving Score|Route Completion|NC " "$L/rt_k6.log" | tail -3
$P tools/plot_replan.py --dump /workspace/data/rt_dump.npz \
    --out /workspace/data/rt_replan.png --panels 0,5,10,15 2>&1 | tail -3
