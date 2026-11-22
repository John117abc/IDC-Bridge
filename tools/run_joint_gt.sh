#!/bin/bash
# 对照实验：外部参考 = 人类 GT 未来（验证注入接口本身是否正常）
set -u
cd /gpudrive/baselines/IDC-Bridge || exit 1
P=/gpudrive/.venv/bin/python
CKPT=/workspace/data/checkpoints/20261119/idc-waymo-v1.0_examples_041839_episode=475.pth
L=/workspace/data/logs
D=/data/data_json/training
W=${W:-12}

$P tools/export_planner_plans.py --data-dir $D --gt --no-filter \
    --files-json /workspace/data/world_files.json \
    --out /workspace/data/gt_plans.npz > "$L/joint_gt_export.log" 2>&1
echo "export exit=$?"

$P src/scripts/eval/idc-eval.py --data-dir $D --model-path "$CKPT" \
    --num-worlds $W --seed 42 \
    --external-candidates /workspace/data/gt_plans.npz \
    --external-now-idx 11 --force-path-index 0 \
    --gif-enabled false > "$L/joint_gt.log" 2>&1
echo "eval exit=$?"
grep -E "Driving Score|Route Completion|NC |TTC" "$L/joint_gt.log" | tail -4
