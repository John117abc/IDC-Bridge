#!/bin/bash
set -u
C=idc_hist
CR=/gpudrive/baselines/IDC-Bridge
PY=/gpudrive/.venv/bin/python
CK=/workspace/data/checkpoints/20261119/idc-waymo-v1.0_examples_041839_episode=475.pth

run() {
  local tag=$1; shift
  docker exec $C bash -lc "cd $CR && timeout 1500 $PY src/scripts/eval/idc-eval.py \
    --data-dir /data/data_json/training --model-path $CK \
    --planner-replan-every 5 --planner-k 1 \
    --num-worlds 6 --epochs 1 --gif-enabled false $* > /tmp/eval_$tag.log 2>&1; echo exit=\$?"
}

echo "=== A) 无规划器基线（6 world）==="
run base
docker exec $C bash -lc "grep -E 'PDMS-rollout' /tmp/eval_base.log | head -8"

echo
echo "=== B) 带规划器（6 world, blend_tail）==="
run rt "--planner-runtime /workspace/data/plant2_ckpt/best.pt"
docker exec $C bash -lc "grep -E 'align|PDMS-rollout' /tmp/eval_rt.log | head -12"
