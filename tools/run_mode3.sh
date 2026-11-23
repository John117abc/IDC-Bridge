#!/bin/bash
set -u
C=idc_hist
CR=/gpudrive/baselines/IDC-Bridge
PY=/gpudrive/.venv/bin/python
CK=/workspace/data/checkpoints/20261119/idc-waymo-v1.0_examples_041839_episode=475.pth

run() {
  local tag=$1; shift
  docker exec $C bash -lc "cd $CR && timeout 900 $PY src/scripts/eval/idc-eval.py \
    --data-dir /data/data_json/training --model-path $CK \
    --num-worlds 4 --epochs 1 --gif-enabled false $* > /tmp/ev_$tag.log 2>&1; echo exit=\$?"
  echo "--- $tag ---"
  docker exec $C bash -lc "grep -E 'PDMS-rollout' /tmp/ev_$tag.log | tail -4"
}

echo "=== A) 无规划器基线 ==="
run a ""
echo "=== B) planner-mode=route（注入 env 自己的参考，验证注入机制）==="
run b "--planner-runtime /workspace/data/plant2_ckpt/best.pt --planner-mode route"
echo "=== C) planner-mode=planner（注入我们的规划器输出）==="
run c "--planner-runtime /workspace/data/plant2_ckpt/best.pt --planner-mode planner"
