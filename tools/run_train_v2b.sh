#!/bin/bash
set -u
C=idc_hist
CR=/gpudrive/baselines/IDC-Bridge
PY=/gpudrive/.venv/bin/python
CK=/workspace/data/checkpoints/20261119/idc-waymo-v1.0_examples_041839_episode=475.pth

echo "=== v2 训练冒烟：2 world × 1 epoch ==="
docker exec $C bash -lc "cd $CR && IDC_STATE_V2=1 IDC_RESET_SCHEDULE=1 timeout 1200 $PY -u src/scripts/train/idc-train.py \
  --data-dir /data/data_json/training --num-worlds 2 --epochs 1 --device cuda \
  --planner-runtime /workspace/data/plant2_ckpt/best.pt \
  --planner-replan-every 5 --planner-k 1 \
  --load-model --model-path $CK --save-freq 1 > /tmp/train_v2b.log 2>&1; echo exit=\$?"

echo "--- 进度 / loss ---"
docker exec $C bash -lc "grep -nE 'TRACE|回合 1/1, 步数 (1|45|91)/91|loss|update|保存|ckpt' /tmp/train_v2b.log | head -12"
echo "--- 末尾 10 行 ---"
docker exec $C bash -lc "tail -10 /tmp/train_v2b.log"
