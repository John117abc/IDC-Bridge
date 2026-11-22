#!/bin/bash
# 多样性消融：2 张 GPU，两波各 2 组，每组 20000 样本 × 8 epoch（WTA+div，不含 IAWR）
set -e
cd /gpudrive/baselines/IDC-Bridge
PY=/gpudrive/.venv/bin/python
D=/workspace/data/planner60k
L=/workspace/data/logs
O=/workspace/data/div
mkdir -p $O

run() {  # $1=名字 $2=gpu 其余=额外参数
  local name=$1 gpu=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu $PY tools/train_planner.py --data $D \
      --out $O/$name --epochs 8 --bs 16 --limit 20000 --lam_iawr 0 \
      "$@" > $L/div_$name.log 2>&1
  CUDA_VISIBLE_DEVICES=$gpu $PY tools/eval_div.py --data $D \
      --ckpt $O/$name/latest.pt --tag $name > $L/div_$name.eval 2>&1
  echo "--- $name done ---"
}

echo "===== 第一波 ====="
run s1_h05 0 --div hinge --div_margin 2.0 --lam_div 0.05 &
P1=$!
run s2_h20 1 --div hinge --div_margin 2.0 --lam_div 0.2 &
P2=$!
wait $P1 $P2

echo "===== 第二波 ====="
run s3_w20 0 --div weighted --lam_div 0.2 &
P3=$!
wait $P3

echo "================ 汇总 ================"
for f in $L/div_*.eval; do cat $f; done
