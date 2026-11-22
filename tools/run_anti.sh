#!/bin/bash
# 60k + IAWR：两种抗模态塌缩机制并行
#   bi = IAWR + 模态专属横向偏置（±2 m 档）
#   ai = IAWR + k-means 锚点初始化 query
set -e
cd /gpudrive/baselines/IDC-Bridge
PY=/gpudrive/.venv/bin/python
D=/workspace/data/planner60k
L=/workspace/data/logs
O=/workspace/data/anti
mkdir -p $O

run() {  # $1=名字 $2=gpu 其余=额外参数
  local name=$1 gpu=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu $PY tools/train_planner.py --data $D \
      --out $O/$name --epochs 8 --bs 16 --reward pdms --workers 8 \
      --lam_iawr 0.1 --div weighted --lam_div 0.01 "$@" > $L/anti_$name.log 2>&1
  CUDA_VISIBLE_DEVICES=$gpu $PY tools/eval_ab_pdms.py --data $D \
      --ckpt $O/$name/latest.pt --limit 200 --out $O/$name/eval.npz \
      > $L/anti_$name.pdms 2>&1
  CUDA_VISIBLE_DEVICES=$gpu $PY tools/eval_div.py --ckpt $O/$name/latest.pt \
      --tag "$name" --pdms-npz $O/$name/eval.npz > $L/anti_$name.eval 2>&1
  echo "--- $name done ---"
}

run bi 0 --iawr_bias 1.0 &
P1=$!
run ai 1 --qinit anchor &
P2=$!
wait $P1 $P2

echo "================ 汇总 ================"
for f in $L/anti_*.eval; do cat $f; done
for f in $L/anti_*.pdms; do grep -E "oracle|全模态|各模态 PDMS 均值" $f; done
