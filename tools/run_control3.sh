#!/bin/bash
set -u
C=idc_hist
CR=/gpudrive/baselines/IDC-Bridge
PY=/gpudrive/.venv/bin/python
CK=/workspace/data/checkpoints/20261119/idc-waymo-v1.0_examples_041839_episode=475.pth

run() {
  local tag=$1; local state=$2; local pl=$3
  docker exec $C bash -lc "cd $CR && IDC_STATE_V2=$state IDC_RESET_SCHEDULE=1 timeout 900 $PY -u src/scripts/train/idc-train.py \
    --data-dir /data/data_json/training --num-worlds 4 --epochs 1 --device cuda \
    --planner-runtime $pl --planner-replan-every 5 --planner-k 1 \
    --load-model --model-path $CK --save-freq 99 > /tmp/ctrl_$tag.log 2>&1; echo exit=\$?"
  echo "--- $tag ---"
  docker exec $C bash -lc "grep -E 'PDMS\]|DIAG-ref' /tmp/ctrl_$tag.log | tail -2"
}

echo "=== 对照 1：v1(66) + 规划器 ==="
run v1 0 /workspace/data/plant2_ckpt/best.pt
echo "=== 对照 2：v2(85) + 规划器（热启动后应与 v1 接近）==="
run v2 1 /workspace/data/plant2_ckpt/best.pt
echo "=== 对照 3：v1(66) 无规划器（基线）==="
run base 0 ""
