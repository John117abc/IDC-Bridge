#!/bin/bash
# P1: 纯 WTA 基线（40 epoch），训练与 P0 修复并行
set -u
cd /gpudrive/baselines/IDC-Bridge || exit 1
P=/gpudrive/.venv/bin/python
D=/workspace/data/planner60k
L=/workspace/data/logs
O=/workspace/data/p1_wta
mkdir -p $O

echo "=== P1-A: WTA only, 40 epochs ==="
CUDA_VISIBLE_DEVICES=0 $P tools/train_planner.py --data $D --out $O \
    --epochs 40 --bs 16 --lam_iawr 0 --lam_div 0.01 --seed 0 \
    > "$L/p1_wta.log" 2>&1
echo "A exit=$?"
tail -3 "$L/p1_wta.log"

echo "=== 评估 A ==="
CUDA_VISIBLE_DEVICES=0 $P tools/eval_ab_pdms.py --data $D \
    --ckpt $O/latest.pt --limit 200 --out $O/eval.npz > "$L/p1_wta_eval.log" 2>&1
CUDA_VISIBLE_DEVICES=0 $P tools/eval_div.py --ckpt $O/latest.pt --tag p1_wta \
    --pdms-npz $O/eval.npz >> "$L/p1_wta_eval.log" 2>&1
tail -12 "$L/p1_wta_eval.log"

echo "=== P1-B: + IAWR 12 epochs（从 A 续训）==="
CUDA_VISIBLE_DEVICES=1 $P tools/train_planner.py --data $D --out /workspace/data/p1_iawr \
    --epochs 12 --bs 16 --reward pdms_gpu --workers 0 --lam_iawr 0.1 --lam_div 0.01 \
    --seed 0 --init-from $O/latest.pt > "$L/p1_iawr.log" 2>&1
echo "B exit=$?"
tail -3 "$L/p1_iawr.log"

echo "=== 评估 B ==="
CUDA_VISIBLE_DEVICES=1 $P tools/eval_ab_pdms.py --data $D \
    --ckpt /workspace/data/p1_iawr/latest.pt --limit 200 \
    --out /workspace/data/p1_iawr/eval.npz > "$L/p1_iawr_eval.log" 2>&1
CUDA_VISIBLE_DEVICES=1 $P tools/eval_div.py --ckpt /workspace/data/p1_iawr/latest.pt \
    --tag p1_iawr --pdms-npz /workspace/data/p1_iawr/eval.npz >> "$L/p1_iawr_eval.log" 2>&1
tail -12 "$L/p1_iawr_eval.log"
echo "=== P1 DONE ==="
