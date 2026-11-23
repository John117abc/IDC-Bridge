#!/bin/bash
set -u
C=idc_hist
CR=/gpudrive/baselines/IDC-Bridge
PY=/gpudrive/.venv/bin/python
CK=/workspace/data/checkpoints/20261119/idc-waymo-v1.0_examples_041839_episode=475.pth

run() {
  local tag=$1; local state=$2; local plarg=$3
  docker exec $C bash -lc "cd $CR && IDC_STATE_V2=$state IDC_RESET_SCHEDULE=1 timeout 900 $PY -u src/scripts/train/idc-train.py \
    --data-dir /data/data_json/training --num-worlds 4 --epochs 1 --device cuda \
    $plarg --planner-replan-every 5 --planner-k 1 \
    --load-model --model-path $CK --save-freq 99 > /tmp/c3_$tag.log 2>&1; echo exit=\$?"
  echo "--- $tag ---"
  docker exec $C bash -lc "grep -E 'PDMS\]' /tmp/c3_$tag.log | tail -1; grep -A4 Traceback /tmp/c3_$tag.log | head -6"
}

echo "=== 1) v1 + 规划器 ==="
run v1p 0 "--planner-runtime /workspace/data/plant2_ckpt/best.pt"
echo "=== 2) v2 + 规划器 ==="
run v2p 1 "--planner-runtime /workspace/data/plant2_ckpt/best.pt"
echo "=== 3) v1 基线（env 自带参考）==="
run v1b 0 "--planner-runtime="
