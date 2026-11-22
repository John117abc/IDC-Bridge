#!/bin/bash
# 用修好的 runtime 出 gif：route（验收） vs planner(K=1)
set -u
cd /gpudrive/baselines/IDC-Bridge || exit 1
P=/gpudrive/.venv/bin/python
CKPT=/workspace/data/checkpoints/20261119/idc-waymo-v1.0_examples_041839_episode=475.pth
L=/workspace/data/logs
D=/data/data_json/training
PL=/workspace/data/ab60k_B/latest.pt
G=/workspace/data/gifs/$(date +%Y%m%d)
mkdir -p $G

echo "### A) route mode (acceptance)"
rm -f $G/story_world*.gif
$P src/scripts/eval/idc-eval.py --data-dir $D --model-path "$CKPT" \
    --num-worlds 12 --seed 42 --planner-runtime "$PL" --planner-mode route \
    --planner-replan-every 5 --gif-enabled true --gif-max-worlds 2 \
    --gif-view-mode bird_2d --gif-world-selection first > "$L/fin_route.log" 2>&1
echo "exit=$?"
grep -E "Driving Score|Route Completion|NC " "$L/fin_route.log" | tail -3
for f in $G/story_world0.gif $G/story_world1.gif; do
  [ -f "$f" ] && cp "$f" "/tmp/fin_route_$(basename $f)"
done

echo "### B) planner mode K=1"
rm -f $G/story_world*.gif
$P src/scripts/eval/idc-eval.py --data-dir $D --model-path "$CKPT" \
    --num-worlds 12 --seed 42 --planner-runtime "$PL" --planner-mode planner \
    --planner-replan-every 5 --planner-k 1 \
    --gif-enabled true --gif-max-worlds 2 \
    --gif-view-mode bird_2d --gif-world-selection first > "$L/fin_planner.log" 2>&1
echo "exit=$?"
grep -E "Driving Score|Route Completion|NC " "$L/fin_planner.log" | tail -3
for f in $G/story_world0.gif $G/story_world1.gif; do
  [ -f "$f" ] && cp "$f" "/tmp/fin_planner_$(basename $f)"
done

ls -la /tmp/fin_*.gif 2>/dev/null
