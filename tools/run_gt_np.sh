#!/bin/bash
# 接口对齐扫描：参考从"现在"开始，扫 now_idx ∈ {41, 0}
set -u
cd /gpudrive/baselines/IDC-Bridge || exit 1
P=/gpudrive/.venv/bin/python
CKPT=/workspace/data/checkpoints/20261119/idc-waymo-v1.0_examples_041839_episode=475.pth
L=/workspace/data/logs
D=/data/data_json/training
W=${W:-12}

$P tools/export_planner_plans.py --data-dir $D --gt-full --no-prefix --no-filter \
    --files-json /workspace/data/world_files.json \
    --out /workspace/data/gt_full_np.npz > "$L/gt_np_export.log" 2>&1
echo "export exit=$? $(tail -1 $L/gt_np_export.log)"

for NOW in 41 0; do
  $P src/scripts/eval/idc-eval.py --data-dir $D --model-path "$CKPT" \
      --num-worlds $W --seed 42 \
      --external-candidates /workspace/data/gt_full_np.npz \
      --external-now-idx $NOW --force-path-index 0 \
      --gif-enabled false > "$L/gt_np_$NOW.log" 2>&1
  echo "--- GT-full, no-prefix, now_idx=$NOW ---"
  grep -E "Driving Score|Route Completion|NC " "$L/gt_np_$NOW.log" | tail -3
done
