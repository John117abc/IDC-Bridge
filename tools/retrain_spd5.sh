#!/bin/bash
set -u
C=idc_hist
CR=/gpudrive/baselines/IDC-Bridge
PY=/gpudrive/.venv/bin/python
CK=/workspace/data/checkpoints/20261119/idc-waymo-v1.0_examples_041839_episode=475.pth
LOG=/workspace/data/logs/train_84_spd5.log

echo "=== 训练前检查：是否有旧训练在跑 ==="
docker exec $C bash -lc "pgrep -af idc-train.py | grep -v pgrep | head -2 || echo 无"

echo "=== 启动重训：84 维 + speed[t+5]，60 epoch × 8 world ==="
docker exec -d $C bash -lc "cd $CR && IDC_STATE_V2=1 IDC_RESET_SCHEDULE=1 $PY -u src/scripts/train/idc-train.py \
  --data-dir /data/data_json/training --num-worlds 8 --epochs 60 --device cuda \
  --planner-runtime= \
  --load-model --model-path $CK --save-freq 10 > $LOG 2>&1"

sleep 150
echo "--- 150 秒后 ---"
docker exec $C bash -lc "grep -c '步数 91/91' $LOG; grep -oE 'PDMS\] score=[0-9.]+' $LOG | tail -5; grep -oE 'DIAG-ref\] median=[0-9.]+' $LOG | tail -2"
