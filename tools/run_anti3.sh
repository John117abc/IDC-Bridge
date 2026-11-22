#!/bin/bash
# GPU 奖励版：从 checkpoint 续训 bi / ai（之前 CPU 回退的墙钟已废弃，权重有效）
set -e
cd /gpudrive/baselines/IDC-Bridge
PY=/gpudrive/.venv/bin/python
D=/workspace/data/planner60k
L=/workspace/data/logs
O=/workspace/data/anti

run() {  # $1=名字 $2=gpu $3=剩余 epoch 其余=额外参数
  local name=$1 gpu=$2 rest=$3; shift 3
  CUDA_VISIBLE_DEVICES=$gpu $PY tools/train_planner.py --data $D \
      --out $O/$name --epochs $rest --bs 16 --reward pdms_gpu --workers 0 \
      --lam_iawr 0.1 --div weighted --lam_div 0.01 \
      --init-from $O/$name/latest.pt "$@" > $L/anti3_$name.log 2>&1
  echo "--- $name 训练完成 ---"
  CUDA_VISIBLE_DEVICES=$gpu $PY tools/eval_ab_pdms.py --data $D \
      --ckpt $O/$name/latest.pt --limit 200 --out $O/$name/eval.npz \
      > $L/anti3_$name.pdms 2>&1
  CUDA_VISIBLE_DEVICES=$gpu $PY tools/eval_div.py --ckpt $O/$name/latest.pt \
      --tag "$name" --pdms-npz $O/$name/eval.npz > $L/anti3_$name.eval 2>&1
  echo "--- $name 评估完成 ---"
}

run bi 0 6 --iawr_bias 1.0 &
P1=$!
run ai 1 5 --qinit anchor &
P2=$!
wait $P1 $P2

echo "================ 汇总 ================"
for name in bi ai; do
  echo "----- $name -----"
  tail -7 $L/anti3_$name.eval
  grep -E "oracle|全模态|各模态 PDMS 均值" $L/anti3_$name.pdms || true
done
