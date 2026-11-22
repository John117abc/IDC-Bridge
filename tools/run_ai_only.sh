#!/bin/bash
# 只续训 ai（qinit 修复后）；bi 已在跑
set -e
cd /gpudrive/baselines/IDC-Bridge
PY=/gpudrive/.venv/bin/python
D=/workspace/data/planner60k
L=/workspace/data/logs
O=/workspace/data/anti

CUDA_VISIBLE_DEVICES=1 $PY tools/train_planner.py --data $D --out $O/ai \
    --epochs 5 --bs 16 --reward pdms --workers 9 --lam_iawr 0.1 \
    --div weighted --lam_div 0.01 --qinit anchor \
    --init-from $O/ai/latest.pt > $L/anti2_ai.log 2>&1

$PY tools/eval_ab_pdms.py --data $D --ckpt $O/ai/latest.pt --limit 200 \
    --out $O/ai/eval.npz > $L/anti2_ai.pdms 2>&1
$PY tools/eval_div.py --ckpt $O/ai/latest.pt --tag ai --pdms-npz $O/ai/eval.npz \
    > $L/anti2_ai.eval 2>&1
echo "===== ai ====="
tail -7 $L/anti2_ai.eval
grep -E "oracle|全模态|各模态 PDMS 均值" $L/anti2_ai.pdms || true
