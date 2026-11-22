#!/bin/bash
# Online runtime acceptance: route mode (re-install env reference) must match baseline;
# then planner mode (K=1).
set -u
cd /gpudrive/baselines/IDC-Bridge || exit 1
P=/gpudrive/.venv/bin/python
CKPT=/workspace/data/checkpoints/20261119/idc-waymo-v1.0_examples_041839_episode=475.pth
L=/workspace/data/logs
D=/data/data_json/training
W=${W:-12}
PL=/workspace/data/ab60k_B/latest.pt

echo "### A) baseline (no injection)"
$P src/scripts/eval/idc-eval.py --data-dir $D --model-path "$CKPT" \
    --num-worlds $W --seed 42 --gif-enabled false > "$L/rt_base.log" 2>&1
echo "exit=$?"
grep -E "Driving Score|Route Completion|NC " "$L/rt_base.log" | tail -3

echo "### B) runtime route mode (acceptance)"
$P src/scripts/eval/idc-eval.py --data-dir $D --model-path "$CKPT" \
    --num-worlds $W --seed 42 --planner-runtime "$PL" --planner-mode route \
    --planner-replan-every 5 --gif-enabled false > "$L/rt_route.log" 2>&1
echo "exit=$?"
grep -E "RT\]|Driving Score|Route Completion|NC " "$L/rt_route.log" | tail -5
grep -iE "error|Traceback" "$L/rt_route.log" | tail -3

echo "### C) runtime planner mode (K=1)"
$P src/scripts/eval/idc-eval.py --data-dir $D --model-path "$CKPT" \
    --num-worlds $W --seed 42 --planner-runtime "$PL" --planner-mode planner \
    --planner-replan-every 5 --planner-k 1 --gif-enabled false > "$L/rt_planner.log" 2>&1
echo "exit=$?"
grep -E "RT\]|Driving Score|Route Completion|NC " "$L/rt_planner.log" | tail -5
grep -iE "error|Traceback" "$L/rt_planner.log" | tail -3
