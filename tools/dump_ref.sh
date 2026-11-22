#!/bin/bash
# 打印 env 内部参考 vs 注入后的参考（world0）
set -u
cd /gpudrive/baselines/IDC-Bridge || exit 1
P=/gpudrive/.venv/bin/python
CKPT=/workspace/data/checkpoints/20261119/idc-waymo-v1.0_examples_041839_episode=475.pth
L=/workspace/data/logs
D=/data/data_json/training
W=${W:-12}

echo "########## 1) 基线（env 内部参考）##########"
$P src/scripts/eval/idc-eval.py --data-dir $D --model-path "$CKPT" \
    --num-worlds $W --seed 42 --dump-ref --gif-enabled false \
    > "$L/dump_base.log" 2>&1
grep -E "DUMP|Driving Score|Route Completion" "$L/dump_base.log" | head -10

echo "########## 2) 注入 GT 参考（replan 通路）##########"
$P src/scripts/eval/idc-eval.py --data-dir $D --model-path "$CKPT" \
    --num-worlds $W --seed 42 --replan-candidates /workspace/data/replan_gt0.npz \
    --force-path-index 0 --external-tail-idx 41 --dump-ref --gif-enabled false \
    > "$L/dump_gt.log" 2>&1
grep -E "DUMP|Driving Score|Route Completion" "$L/dump_gt.log" | head -10
