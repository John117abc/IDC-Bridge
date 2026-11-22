#!/bin/bash
# 修复 densify 双调用后重跑：GT 单帧参考（带重锚定 + 尾段接回 + maps）
set -u
cd /gpudrive/baselines/IDC-Bridge || exit 1
P=/gpudrive/.venv/bin/python
CKPT=/workspace/data/checkpoints/20261119/idc-waymo-v1.0_examples_041839_episode=475.pth
L=/workspace/data/logs
D=/data/data_json/training
W=${W:-12}

$P tools/export_replan.py --data-dir $D --gt --frames 0 --k 1 \
    --files-json /workspace/data/world_files.json \
    --out /workspace/data/replan_gt0.npz > "$L/replan_gt0_export.log" 2>&1
echo "export exit=$? $(tail -1 $L/replan_gt0_export.log)"

echo "=== GT 单帧 + 重锚定 + 尾段接回 ==="
$P src/scripts/eval/idc-eval.py --data-dir $D --model-path "$CKPT" \
    --num-worlds $W --seed 42 --replan-candidates /workspace/data/replan_gt0.npz \
    --force-path-index 0 --external-blend-tail --external-tail-idx 41 \
    --dump-ref --gif-enabled false > "$L/fix_gt.log" 2>&1
grep -E "road_dist|Driving Score|Route Completion|NC " "$L/fix_gt.log" | tail -4
echo "=== 基线 ==="
grep -E "Driving Score|Route Completion|NC " "$L/blend_base.log" | tail -3
