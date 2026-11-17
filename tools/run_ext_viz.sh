#!/usr/bin/env bash
set -u
cd /gpudrive/baselines/IDC-Bridge || exit 1
P=/gpudrive/.venv/bin/python
CKPT=/workspace/data/checkpoints/20261115/idc-waymo-v1.0_examples_201537_episode=3000.pth
L=/workspace/data/logs

rm -rf /workspace/data/gifs/extviz
mkdir -p /workspace/data/gifs/extviz

$P src/scripts/eval/idc-eval.py --data-dir /data/data_json/training \
    --model-path "$CKPT" --num-worlds 12 --seed 42 \
    --external-candidates /workspace/data/head_plans2.npz \
    --gif-enabled true --gif-max-worlds 3 --gif-view-mode bird_2d \
    --gif-world-selection first > "$L/ext_viz.log" 2>&1
echo "eval exit=$?" >> "$L/ext_viz.log"
cp /workspace/data/gifs/*/story_world*.gif /workspace/data/gifs/extviz/ 2>/dev/null
cp /workspace/data/gifs/*/story_world*.png /workspace/data/gifs/extviz/ 2>/dev/null
ls /workspace/data/gifs/extviz > "$L/ext_viz_list.txt"
grep -E "Driving Score|Route Completion|NC " "$L/ext_viz.log" | tail -3 >> "$L/ext_viz.log"
echo DONE >> "$L/ext_viz.log"
