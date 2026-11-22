#!/bin/bash
# 混合参考验证：4 s 规划 + 之后接回路线（blend_tail）
set -u
cd /gpudrive/baselines/IDC-Bridge || exit 1
P=/gpudrive/.venv/bin/python
CKPT=/workspace/data/checkpoints/20261119/idc-waymo-v1.0_examples_041839_episode=475.pth
L=/workspace/data/logs
D=/data/data_json/training
W=${W:-12}

echo "=== A) GT 参考 + blend_tail（接口验收）==="
$P src/scripts/eval/idc-eval.py --data-dir $D --model-path "$CKPT" \
    --num-worlds $W --seed 42 --replan-candidates /workspace/data/replan_gt0.npz \
    --force-path-index 0 --external-blend-tail --external-tail-idx 41 \
    --gif-enabled false > "$L/blend_gt.log" 2>&1
grep -E "Driving Score|Route Completion|NC " "$L/blend_gt.log" | tail -3

echo "=== B) 基线（无注入）==="
$P src/scripts/eval/idc-eval.py --data-dir $D --model-path "$CKPT" \
    --num-worlds $W --seed 42 --gif-enabled false > "$L/blend_base.log" 2>&1
grep -E "Driving Score|Route Completion|NC " "$L/blend_base.log" | tail -3
