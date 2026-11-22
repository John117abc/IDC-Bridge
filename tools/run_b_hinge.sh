#!/bin/bash
# 60k + IAWR + hinge 多样性：λ_div 两档并行，看能否在保留 IAWR 收益的同时恢复模态差异
set -e
cd /gpudrive/baselines/IDC-Bridge
PY=/gpudrive/.venv/bin/python
D=/workspace/data/planner60k
L=/workspace/data/logs
O=/workspace/data/bh
mkdir -p $O

run() {  # $1=名字 $2=gpu $3=lam_div
  local name=$1 gpu=$2 lam=$3
  CUDA_VISIBLE_DEVICES=$gpu $PY tools/train_planner.py --data $D \
      --out $O/$name --epochs 8 --bs 16 --reward pdms --workers 8 \
      --lam_iawr 0.1 --div hinge --div_margin 2.0 --lam_div $lam \
      > $L/bh_$name.log 2>&1
  CUDA_VISIBLE_DEVICES=$gpu $PY tools/eval_div.py --ckpt $O/$name/latest.pt \
      --tag "$name (60k+IAWR)" > $L/bh_$name.eval 2>&1
  CUDA_VISIBLE_DEVICES=$gpu $PY tools/eval_ab_pdms.py --data $D \
      --ckpt $O/$name/latest.pt --limit 200 --out $O/$name/eval.npz \
      > $L/bh_$name.pdms 2>&1
  echo "--- $name done ---"
}

run bh05 0 0.05 &
P1=$!
run bh10 1 0.10 &
P2=$!
wait $P1 $P2

echo "================ 汇总 ================"
for f in $L/bh_*.eval; do cat $f; done
for f in $L/bh_*.pdms; do grep -E "oracle|全模态|各模态 PDMS 均值" $f; done
