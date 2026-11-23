#!/bin/bash
set -u
C=idc_hist
CR=/gpudrive/baselines/IDC-Bridge
PY=/gpudrive/.venv/bin/python
CK=/workspace/data/checkpoints/20261119/idc-waymo-v1.0_examples_041839_episode=475.pth

docker exec $C bash -lc "cd $CR && $PY tools/patch_dbg_ref.py 2>&1 | tail -2"

echo "=== 跑 10 步看参考推进 ==="
docker exec $C bash -lc "cd $CR && IDC_STATE_V2=1 IDC_RESET_SCHEDULE=1 timeout 600 $PY -u src/scripts/train/idc-train.py \
  --data-dir /data/data_json/training --num-worlds 2 --epochs 1 --device cuda \
  --planner-runtime /workspace/data/plant2_ckpt/best.pt \
  --planner-replan-every 5 --planner-k 1 \
  --load-model --model-path $CK --save-freq 99 > /tmp/train_dbg.log 2>&1; echo exit=\$?"

echo "--- DBG-ref ---"
docker exec $C bash -lc "grep -E 'DBG-ref|DIAG-ref' /tmp/train_dbg.log | head -16"
