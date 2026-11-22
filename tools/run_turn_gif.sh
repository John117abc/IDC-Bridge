#!/bin/bash
# Turning-world comparison: route vs planner for one world id (WID)
set -u
cd /gpudrive/baselines/IDC-Bridge || exit 1
P=/gpudrive/.venv/bin/python
CKPT=/workspace/data/checkpoints/20261119/idc-waymo-v1.0_examples_041839_episode=475.pth
L=/workspace/data/logs
D=/data/data_json/training
PL=/workspace/data/ab60k_B/latest.pt
G=/workspace/data/gifs/$(date +%Y%m%d)
WID=${WID:-2}
mkdir -p $G

echo "### route mode, world $WID"
rm -f $G/story_world*.gif
$P src/scripts/eval/idc-eval.py --data-dir $D --model-path "$CKPT" \
    --num-worlds 12 --seed 42 --planner-runtime "$PL" --planner-mode route \
    --planner-replan-every 5 --gif-enabled true --gif-worlds "$WID" \
    --gif-view-mode bird_2d > "$L/tw${WID}_route.log" 2>&1
echo "exit=$?"
grep -E "Driving Score|Route Completion" "$L/tw${WID}_route.log" | tail -2
if [ -f "$G/story_world${WID}.gif" ]; then cp "$G/story_world${WID}.gif" "/tmp/w${WID}_route.gif"; echo "route gif ok"; fi

echo "### planner mode K=1, world $WID"
rm -f $G/story_world*.gif
$P src/scripts/eval/idc-eval.py --data-dir $D --model-path "$CKPT" \
    --num-worlds 12 --seed 42 --planner-runtime "$PL" --planner-mode planner \
    --planner-replan-every 5 --planner-k 1 --gif-enabled true --gif-worlds "$WID" \
    --gif-view-mode bird_2d > "$L/tw${WID}_planner.log" 2>&1
echo "exit=$?"
grep -E "Driving Score|Route Completion" "$L/tw${WID}_planner.log" | tail -2
if [ -f "$G/story_world${WID}.gif" ]; then cp "$G/story_world${WID}.gif" "/tmp/w${WID}_planner.gif"; echo "planner gif ok"; fi

ls -la "/tmp/w${WID}_route.gif" "/tmp/w${WID}_planner.gif" 2>/dev/null
