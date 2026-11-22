#!/bin/bash
# 定责实验：只给 IDC 看规划器"按 GT 排序后的第 1 条"，强制走它
set -u
cd /gpudrive/baselines/IDC-Bridge || exit 1
P=/gpudrive/.venv/bin/python
CKPT=/workspace/data/checkpoints/20261119/idc-waymo-v1.0_examples_041839_episode=475.pth
L=/workspace/data/logs
D=/data/data_json/training
W=${W:-12}

$P tools/export_planner_plans.py --data-dir $D \
    --ckpt /workspace/data/ab60k_B/latest.pt --oracle --no-filter \
    --files-json /workspace/data/world_files.json \
    --out /workspace/data/planner_plans_oracle.npz > "$L/joint_oracle_export.log" 2>&1
echo "export exit=$?"

$P src/scripts/eval/idc-eval.py --data-dir $D --model-path "$CKPT" \
    --num-worlds $W --seed 42 \
    --external-candidates /workspace/data/planner_plans_oracle.npz \
    --external-now-idx 11 --force-path-index 0 \
    --gif-enabled true --gif-max-worlds 3 --gif-view-mode bird_2d \
    --gif-world-selection first > "$L/joint_oracle.log" 2>&1
echo "eval exit=$?"
grep -E "Driving Score|Route Completion|NC |TTC" "$L/joint_oracle.log" | tail -4
