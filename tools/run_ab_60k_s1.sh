#!/bin/bash
# seed 1 复核：A/B 各 8 epoch + 评估
set -e
cd /gpudrive/baselines/IDC-Bridge
PY=/gpudrive/.venv/bin/python
D=/workspace/data/planner60k
L=/workspace/data/logs
EP=${EP:-8}; BS=${BS:-16}; W=${W:-12}; SEED=${SEED:-1}
A=/workspace/data/ab60k_A_s$SEED
B=/workspace/data/ab60k_B_s$SEED

echo "===== A seed=$SEED ====="
$PY tools/train_planner.py --data $D --out $A --epochs $EP --bs $BS --seed $SEED \
    --lam_iawr 0 2>&1 | tee $L/ab60k_A_s$SEED.log

echo "===== B seed=$SEED ====="
$PY tools/train_planner.py --data $D --out $B --epochs $EP --bs $BS --seed $SEED \
    --reward pdms --workers $W --lam_iawr 0.1 2>&1 | tee $L/ab60k_B_s$SEED.log

echo "===== eval ====="
$PY tools/eval_ab_pdms.py --data $D --ckpt $A/latest.pt --limit 200 \
    --out $A/eval.npz 2>&1 | tail -12
$PY tools/eval_ab_pdms.py --data $D --ckpt $B/latest.pt --limit 200 \
    --out $B/eval.npz 2>&1 | tail -12
echo "===== A/B stats seed=$SEED ====="
$PY tools/ab_stats.py --a $A/eval.npz --b $B/eval.npz 2>&1 | tail -25
echo DONE_S$SEED
