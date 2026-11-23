#!/bin/bash
set -u
C=idc_hist
CR=/gpudrive/baselines/IDC-Bridge
PY=/gpudrive/.venv/bin/python
CK=/workspace/data/checkpoints/20261119/idc-waymo-v1.0_examples_041839_episode=475.pth

echo "=== v2(84) 基线：与训练流程一致，仅换输入（无规划器注入）==="
docker exec $C bash -lc "cd $CR && IDC_STATE_V2=1 IDC_RESET_SCHEDULE=1 timeout 900 $PY -u src/scripts/train/idc-train.py \
  --data-dir /data/data_json/training --num-worlds 4 --epochs 1 --device cuda \
  --planner-runtime= \
  --load-model --model-path $CK --save-freq 99 > /tmp/v84_base.log 2>&1; echo exit=\$?"
echo "--- 结果 ---"
docker exec $C bash -lc "grep -E 'PDMS\]|维度|热启动|Traceback' /tmp/v84_base.log | tail -6"
