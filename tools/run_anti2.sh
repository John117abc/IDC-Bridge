#!/bin/bash
# 从各自 checkpoint 续训（之前因容器 GPU 掉落跑到 CPU 上，已废弃这些 epoch 的墙钟）
set -e
cd /gpudrive/baselines/IDC-Bridge
PY=/gpudrive/.venv/bin/python
D=/workspace/data/planner60k
L=/workspace/data/logs
O=/workspace/data/anti

run() {  # $1=名字 $2=gpu $3=续训起点的 epoch 数 其余=额外参数
  local name=$1 gpu=$2; shift 2
  local ep_rest=${EPREST:-6}
  CUDA_VISIBLE_DEVICES=$gpu $PY tools/train_planner.py --data $D \
      --out $O/$name --epochs $ep_rest --bs 16 --reward pdms --workers 9 \
      --lam_iawr 0.1 --div weighted --lam_div 0.01 \
      --init-from $O/$name/latest.pt "$@" > $L/anti2_$name.log 2>&1
  echo "--- $name 续训完成 ---"
}

EPREST=6 run bi 0 --iawr_bias 1.0 &
P1=$!
EPREST=5 run ai 1 --qinit anchor &
P2=$!
wait $P1 $P2

echo "================ 评估 ================"
for name in bi ai; do
  $PY tools/eval_ab_pdms.py --data $D --ckpt $O/$name/latest.pt --limit 200 \
      --out $O/$name/eval.npz > $L/anti2_$name.pdms 2>&1
  $PY tools/eval_div.py --ckpt $O/$name/latest.pt --tag "$name" \
      --pdms-npz $O/$name/eval.npz > $L/anti2_$name.eval 2>&1
  echo "----- $name -----"
  tail -7 $L/anti2_$name.eval
  grep -E "oracle|全模态|各模态 PDMS 均值" $L/anti2_$name.pdms || true
done
