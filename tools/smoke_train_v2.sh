#!/bin/bash
set -u
C=idc_hist
CR=/gpudrive/baselines/IDC-Bridge
PY=/gpudrive/.venv/bin/python
CK=/workspace/data/checkpoints/20261119/idc-waymo-v1.0_examples_041839_episode=475.pth

echo "=== v2 训练冒烟：1 epoch × 4 world ==="
docker exec $C bash -lc "cd $CR && IDC_STATE_V2=1 timeout 1800 $PY src/scripts/train/idc-train.py \
  --data-dir /data/data_json/training \
  --num-worlds 4 --epochs 1 --device cuda \
  --planner-runtime /workspace/data/plant2_ckpt/best.pt \
  --planner-replan-every 5 --planner-k 1 \
  --load-model --model-path $CK \
  --save-freq 1 > /tmp/train_v2.log 2>&1; echo exit=\$?"

echo "--- 关键日志 ---"
docker exec $C bash -lc "grep -E 'dim|维度|PHASE|RT\]|loss|Loss|Traceback|跳过' /tmp/train_v2.log | head -16"
echo "--- 错误尾部 ---"
docker exec $C bash -lc "grep -A8 'Traceback' /tmp/train_v2.log | head -16"
echo "--- 末 6 行 ---"
docker exec $C bash -lc "tail -6 /tmp/train_v2.log"
