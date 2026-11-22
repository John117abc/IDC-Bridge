#!/bin/bash
# Route-mode gifs for several worlds at once (to pick ones with obvious turns).
set -u
cd /gpudrive/baselines/IDC-Bridge || exit 1
P=/gpudrive/.venv/bin/python
CKPT=/workspace/data/checkpoints/20261119/idc-waymo-v1.0_examples_041839_episode=475.pth
L=/workspace/data/logs
D=/data/data_json/training
PL=/workspace/data/p1_wta/latest.pt
G=/workspace/data/gifs/$(date +%Y%m%d)
IDS=${IDS:-0,1,2,4,6,9,10,11}
MODE=${MODE:-route}
mkdir -p $G

rm -f $G/story_world*.gif
$P src/scripts/eval/idc-eval.py --data-dir $D --model-path "$CKPT" \
    --num-worlds 12 --seed 42 --planner-runtime "$PL" --planner-mode "$MODE" \
    --planner-replan-every 5 --planner-k 1 \
    --gif-enabled true --gif-worlds "$IDS" --gif-view-mode bird_2d \
    --traj-plot-max-worlds 12 > "$L/multi_${MODE}.log" 2>&1
echo "exit=$?"
grep -E "Driving Score|Route Completion" "$L/multi_${MODE}.log" | tail -2

for w in $(echo $IDS | tr ',' ' '); do
  if [ -f "$G/story_world${w}.gif" ]; then
    cp "$G/story_world${w}.gif" "/tmp/mw${w}_${MODE}.gif"
    echo "copied world $w"
  fi
done
ls /tmp/mw*_${MODE}.gif 2>/dev/null | wc -l
