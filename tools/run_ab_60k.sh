#!/bin/bash
# 60k 上同一预算的 A/B：A = WTA+多样性；B = A + IAWR（PDMS 奖励）
set -e
cd /gpudrive/baselines/IDC-Bridge
PY=/gpudrive/.venv/bin/python
D=/workspace/data/planner60k
L=/workspace/data/logs
EP=${EP:-8}
BS=${BS:-16}
W=${W:-12}

echo "===== A: WTA + 多样性 (epochs=$EP bs=$BS) ====="
$PY tools/train_planner.py --data $D --out /workspace/data/ab60k_A \
    --epochs $EP --bs $BS --lam_iawr 0 2>&1 | tee $L/ab60k_A.log

echo "===== B: + IAWR (PDMS 奖励, workers=$W) ====="
$PY tools/train_planner.py --data $D --out /workspace/data/ab60k_B \
    --epochs $EP --bs $BS --reward pdms --workers $W --lam_iawr 0.1 \
    2>&1 | tee $L/ab60k_B.log

echo "===== 完成 ====="
