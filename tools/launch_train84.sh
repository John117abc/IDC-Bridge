#!/bin/bash
set -u
C=idc_hist
CR=/gpudrive/baselines/IDC-Bridge
PY=/gpudrive/.venv/bin/python
CK=/workspace/data/checkpoints/20261119/idc-waymo-v1.0_examples_041839_episode=475.pth
LOG=/workspace/data/logs/train_84_long.log

echo "=== 启动正式训练：84 维输入，基线同款流程，无规划器注入 ==="
docker exec -d $C bash -lc "cd $CR && IDC_STATE_V2=1 IDC_RESET_SCHEDULE=1 $PY -u src/scripts/train/idc-train.py \
  --data-dir /data/data_json/training --num-worlds 8 --epochs 30 --device cuda \
  --planner-runtime= \
  --load-model --model-path $CK --save-freq 5 > $LOG 2>&1"

sleep 45
echo "--- 45 秒后进度 ---"
docker exec $C bash -lc "grep -cE '步数' $LOG; grep -E 'PDMS\]|保存模型|Traceback' $LOG | tail -5"
echo "--- 进程 ---"
docker exec $C bash -lc "pgrep -af 'idc-train.py' | grep -v pgrep | head -2"
