#!/bin/bash
# 用在线 runtime 生成 gif：route 模式（正常参考） vs planner 模式（规划器参考）
set -u
cd /gpudrive/baselines/IDC-Bridge || exit 1
P=/gpudrive/.venv/bin/python
CKPT=/workspace/data/checkpoints/20261119/idc-waymo-v1.0_examples_041839_episode=475.pth
L=/workspace/data/logs
D=/data/data_json/training
W=${W:-12}
PL=/workspace/data/ab60k_B/latest.pt
G=/workspace/data/gifs/$(date +%Y%m%d)

echo "### route mode + gif"
$P src/scripts/eval/idc-eval.py --data-dir $D --model-path "$CKPT" \
    --num-worlds $W --seed 42 --planner-runtime "$PL" --planner-mode route \
    --planner-replan-every 5 --gif-enabled true --gif-max-worlds 2 \
    --gif-view-mode bird_2d --gif-world-selection first > "$L/gif_route.log" 2>&1
echo "exit=$?"
grep -E "Driving Score|Route Completion" "$L/gif_route.log" | tail -2
for f in "$G"/story_world*.gif; do cp "$f" "/tmp/route_$(basename $f)" 2>/dev/null; done

echo "### planner mode + gif"
$P src/scripts/eval/idc-eval.py --data-dir $D --model-path "$CKPT" \
    --num-worlds $W --seed 42 --planner-runtime "$PL" --planner-mode planner \
    --planner-replan-every 5 --planner-k 1 \
    --gif-enabled true --gif-max-worlds 2 \
    --gif-view-mode bird_2d --gif-world-selection first > "$L/gif_planner.log" 2>&1
echo "exit=$?"
grep -E "Driving Score|Route Completion" "$L/gif_planner.log" | tail -2
for f in "$G"/story_world*.gif; do cp "$f" "/tmp/planner_$(basename $f)" 2>/dev/null; done

ls -la /tmp/route_*.gif /tmp/planner_*.gif 2>/dev/null
