#!/bin/bash
# 修复队列：杀掉残留 -> 评估 A -> P1-B(IAWR) -> P2(锚点 40ep + IAWR 12ep) -> 全部评估
set -u
cd /gpudrive/baselines/IDC-Bridge || exit 1
P=/gpudrive/.venv/bin/python
D=/workspace/data/planner60k
L=/workspace/data/logs
A=/workspace/data/p1_wta
AI=/workspace/data/p1_iawr
BO=/workspace/data/p2_anchor
BI=/workspace/data/p2_anchor_iawr

pkill -f run_p1.sh 2>/dev/null
pkill -f run_p2.sh 2>/dev/null
pkill -f train_planner 2>/dev/null
sleep 2
echo "残留进程: $(pgrep -f train_planner | wc -l)"

echo "=== 0) 评估 A（自由模态 40ep）==="
$P tools/eval_ab_pdms.py --data $D --ckpt $A/latest.pt --limit 200 \
    --out $A/eval.npz > "$L/p1_wta_eval.log" 2>&1
$P tools/eval_div.py --ckpt $A/latest.pt --tag p1_wta --pdms-npz $A/eval.npz \
    >> "$L/p1_wta_eval.log" 2>&1
tail -10 "$L/p1_wta_eval.log"

echo "=== 1) P1-B: A + IAWR 12ep ==="
CUDA_VISIBLE_DEVICES=0 $P tools/train_planner.py --data $D --out $AI \
    --epochs 12 --bs 16 --reward pdms_gpu --workers 0 --lam_iawr 0.1 \
    --lam_div 0.01 --seed 0 --init-from $A/latest.pt > "$L/p1_iawr.log" 2>&1
echo "exit=$?"; tail -3 "$L/p1_iawr.log"
$P tools/eval_ab_pdms.py --data $D --ckpt $AI/latest.pt --limit 200 \
    --out $AI/eval.npz > "$L/p1_iawr_eval.log" 2>&1
$P tools/eval_div.py --ckpt $AI/latest.pt --tag p1_iawr --pdms-npz $AI/eval.npz \
    >> "$L/p1_iawr_eval.log" 2>&1
tail -10 "$L/p1_iawr_eval.log"

echo "=== 2) P2-A: 锚点模态 40ep ==="
CUDA_VISIBLE_DEVICES=0 $P tools/train_planner.py --data $D --out $BO \
    --epochs 40 --bs 16 --lam_iawr 0 --lam_div 0.01 --seed 0 \
    --anchor-modes --res-scale 3.0 > "$L/p2_anchor.log" 2>&1
echo "exit=$?"; tail -3 "$L/p2_anchor.log"
$P tools/eval_ab_pdms.py --data $D --ckpt $BO/latest.pt --limit 200 \
    --out $BO/eval.npz > "$L/p2_anchor_eval.log" 2>&1
$P tools/eval_div.py --ckpt $BO/latest.pt --tag p2_anchor --pdms-npz $BO/eval.npz \
    >> "$L/p2_anchor_eval.log" 2>&1
tail -10 "$L/p2_anchor_eval.log"

echo "=== 3) P2-B: 锚点 + IAWR 12ep ==="
CUDA_VISIBLE_DEVICES=1 $P tools/train_planner.py --data $D --out $BI \
    --epochs 12 --bs 16 --reward pdms_gpu --workers 0 --lam_iawr 0.1 \
    --lam_div 0.01 --seed 0 --anchor-modes --res-scale 3.0 \
    --init-from $BO/latest.pt > "$L/p2_anchor_iawr.log" 2>&1
echo "exit=$?"; tail -3 "$L/p2_anchor_iawr.log"
$P tools/eval_ab_pdms.py --data $D --ckpt $BI/latest.pt --limit 200 \
    --out $BI/eval.npz > "$L/p2_anchor_iawr_eval.log" 2>&1
$P tools/eval_div.py --ckpt $BI/latest.pt --tag p2_anchor_iawr --pdms-npz $BI/eval.npz \
    >> "$L/p2_anchor_iawr_eval.log" 2>&1
tail -10 "$L/p2_anchor_iawr_eval.log"

echo "=== ALL DONE ==="
