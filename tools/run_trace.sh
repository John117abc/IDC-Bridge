#!/bin/bash
set -u
C=idc_hist
CR=/gpudrive/baselines/IDC-Bridge
PY=/gpudrive/.venv/bin/python
CK=/workspace/data/checkpoints/20261119/idc-waymo-v1.0_examples_041839_episode=475.pth

echo "=== 直接跑（前台，看 trace）==="
docker exec $C bash -lc "cd $CR && IDC_STATE_V2=1 timeout 240 $PY -u src/scripts/train/idc-train.py \
  --data-dir /data/data_json/training --num-worlds 2 --epochs 1 --device cuda \
  --planner-runtime /workspace/data/plant2_ckpt/best.pt \
  --planner-replan-every 5 --planner-k 1 \
  --load-model --model-path $CK --save-freq 1 > /tmp/train_trace.log 2>&1; echo exit=\$?"
echo "--- TRACE 与结尾 ---"
docker exec $C bash -lc "grep -nE 'TRACE|Traceback|Error|Error:' /tmp/train_trace.log | head -12; echo ...; tail -6 /tmp/train_trace.log"
