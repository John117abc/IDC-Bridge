#!/usr/bin/env bash
# 对比两个执行器 checkpoint：同一批 world、固定 gif 世界，分别保存 GIF
set -u
cd /gpudrive/baselines/IDC-Bridge || exit 1
P=/gpudrive/.venv/bin/python
L=/workspace/data/logs
CMP=/workspace/data/gifs/cmp
mkdir -p "$CMP"

run_one () {
    NAME=$1
    CKPT=$2
    rm -f /workspace/data/gifs/*/story_world*.gif /workspace/data/gifs/*/story_world*.png 2>/dev/null
    $P src/scripts/eval/idc-eval.py --data-dir /data/data_json/training \
        --model-path "$CKPT" --num-worlds 12 --seed 42 \
        --gif-enabled true --gif-max-worlds 3 --gif-view-mode bird_2d \
        --gif-world-selection first > "$L/cmp_${NAME}.log" 2>&1
    grep -E "Driving Score|Route Completion|NC " "$L/cmp_${NAME}.log" | tail -3 \
        >> "$L/cmp_summary.log"
    for f in /workspace/data/gifs/*/story_world*.gif; do
        [ -e "$f" ] || continue
        b=$(basename "$f" .gif)
        cp "$f" "$CMP/${NAME}_${b}.gif"
    done
    echo "[cmp] $NAME done" >> "$L/cmp_summary.log"
}

echo "===== [CMP] start $(date '+%F %T') =====" > "$L/cmp_summary.log"
run_one ep3000 /workspace/data/checkpoints/20261115/idc-waymo-v1.0_examples_201537_episode=3000.pth
run_one ep360 /workspace/data/checkpoints/20261116/idc-waymo-v1.0_examples_083455_episode=360.pth
echo "===== [CMP] DONE $(date '+%F %T') =====" >> "$L/cmp_summary.log"
