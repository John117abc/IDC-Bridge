#!/bin/bash
# 用新的"互不相同且可用"指标回算已有 checkpoint（A / B / 两个 B+hinge）
set -e
cd /gpudrive/baselines/IDC-Bridge
PY=/gpudrive/.venv/bin/python

echo "===== A (60k) ====="
$PY tools/eval_div.py --ckpt /workspace/data/ab60k_A/latest.pt --tag "A 60k" \
    --pdms-npz /workspace/data/ab60k_A/eval.npz 2>/dev/null | tail -4
echo "===== B (60k, +IAWR) ====="
$PY tools/eval_div.py --ckpt /workspace/data/ab60k_B/latest.pt --tag "B 60k" \
    --pdms-npz /workspace/data/ab60k_B/eval.npz 2>/dev/null | tail -4
echo "===== B + hinge 0.05 ====="
$PY tools/eval_div.py --ckpt /workspace/data/bh/bh05/latest.pt --tag "bh05" \
    --pdms-npz /workspace/data/bh/bh05/eval.npz 2>/dev/null | tail -4 || true
echo "===== B + hinge 0.10 ====="
$PY tools/eval_div.py --ckpt /workspace/data/bh/bh10/latest.pt --tag "bh10" \
    --pdms-npz /workspace/data/bh/bh10/eval.npz 2>/dev/null | tail -4 || true
