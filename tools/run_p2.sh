#!/bin/bash
# 等 P1（A + IAWR）跑完，然后跑方案 B（锚点模态）并评估
set -u
cd /gpudrive/baselines/IDC-Bridge || exit 1
P=/gpudrive/.venv/bin/python
D=/workspace/data/planner60k
L=/workspace/data/logs
BO=/workspace/data/p2_anchor

echo "=== 等待 P1 结束 ==="
for i in $(seq 1 240); do
  n=$(pgrep -f run_p1.sh | wc -l)
  if [ "$n" -eq 0 ]; then break; fi
  sleep 60
done
echo "P1 已结束（或超时），开始 B"

mkdir -p $BO
echo "=== B-A: 锚点模态 + WTA，30 epoch ==="
CUDA_VISIBLE_DEVICES=0 $P tools/train_planner.py --data $D --out $BO \
    --epochs 40 --bs 16 --lam_iawr 0 --lam_div 0.01 --seed 0 \
    --anchor-modes --res-scale 3.0 > "$L/p2_anchor.log" 2>&1
echo "B-A exit=$?"
tail -4 "$L/p2_anchor.log"

echo "=== 评估 B-A ==="
CUDA_VISIBLE_DEVICES=0 $P tools/eval_ab_pdms.py --data $D --ckpt $BO/latest.pt \
    --limit 200 --out $BO/eval.npz > "$L/p2_anchor_eval.log" 2>&1
CUDA_VISIBLE_DEVICES=0 $P tools/eval_div.py --ckpt $BO/latest.pt --tag p2_anchor \
    --pdms-npz $BO/eval.npz >> "$L/p2_anchor_eval.log" 2>&1
tail -14 "$L/p2_anchor_eval.log"

echo "=== B-B: + IAWR 12 epoch（锚点残差）==="
CUDA_VISIBLE_DEVICES=1 $P tools/train_planner.py --data $D --out ${BO}_iawr \
    --epochs 12 --bs 16 --reward pdms_gpu --workers 0 --lam_iawr 0.1 --lam_div 0.01 \
    --seed 0 --anchor-modes --res-scale 3.0 --init-from $BO/latest.pt \
    > "$L/p2_anchor_iawr.log" 2>&1
echo "B-B exit=$?"
tail -4 "$L/p2_anchor_iawr.log"

echo "=== 评估 B-B ==="
CUDA_VISIBLE_DEVICES=1 $P tools/eval_ab_pdms.py --data $D --ckpt ${BO}_iawr/latest.pt \
    --limit 200 --out ${BO}_iawr/eval.npz > "$L/p2_anchor_iawr_eval.log" 2>&1
CUDA_VISIBLE_DEVICES=1 $P tools/eval_div.py --ckpt ${BO}_iawr/latest.pt --tag p2_anchor_iawr \
    --pdms-npz ${BO}_iawr/eval.npz >> "$L/p2_anchor_iawr_eval.log" 2>&1
tail -14 "$L/p2_anchor_iawr_eval.log"
echo "=== P2 DONE ==="
