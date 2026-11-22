#!/bin/bash
# 1) 60k 预算下 A/B 两组的多样性；2) 多样性消融两组的 oracle PDMS
set -e
cd /gpudrive/baselines/IDC-Bridge
PY=/gpudrive/.venv/bin/python
D=/workspace/data/planner60k

echo "########## 60k 训练出的模型（54000 样本 × 8 epoch）##########"
for c in ab60k_A ab60k_B; do
  $PY tools/eval_div.py --ckpt /workspace/data/$c/latest.pt --tag "$c (60k)" 2>/dev/null | tail -6
done

echo "########## 多样性消融（20000 样本 × 8 epoch）的 oracle PDMS ##########"
for c in s0_base s1_h05 s2_h20 s3_w20; do
  $PY tools/eval_ab_pdms.py --data $D --ckpt /workspace/data/div/$c/latest.pt \
      --limit 200 --out /workspace/data/div/$c/eval.npz 2>/dev/null \
      | grep -E "oracle|全模态|各模态 PDMS 均值|按 minFDE"
done
