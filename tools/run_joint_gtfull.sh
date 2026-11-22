#!/bin/bash
# 最干净的接口对照：完整 91 点人类轨迹当外部候选
set -u
cd /gpudrive/baselines/IDC-Bridge || exit 1
P=/gpudrive/.venv/bin/python
CKPT=/workspace/data/checkpoints/20261119/idc-waymo-v1.0_examples_041839_episode=475.pth
L=/workspace/data/logs
D=/data/data_json/training
W=${W:-12}

$P tools/export_planner_plans.py --data-dir $D --gt-full --no-filter \
    --files-json /workspace/data/world_files.json \
    --out /workspace/data/gt_full.npz > "$L/joint_gtfull_export.log" 2>&1
echo "export exit=$? $(tail -1 $L/joint_gtfull_export.log)"

for NOW in 0 11; do
  $P src/scripts/eval/idc-eval.py --data-dir $D --model-path "$CKPT" \
      --num-worlds $W --seed 42 \
      --external-candidates /workspace/data/gt_full.npz \
      --external-now-idx $NOW --force-path-index 0 \
      --gif-enabled false > "$L/joint_gtfull_$NOW.log" 2>&1
  echo "--- now_idx=$NOW ---"
  grep -E "Driving Score|Route Completion|NC " "$L/joint_gtfull_$NOW.log" | tail -3
done
