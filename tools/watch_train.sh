#!/bin/bash
set -u
C=idc_hist
CR=/gpudrive/baselines/IDC-Bridge
PY=/gpudrive/.venv/bin/python
CK=/workspace/data/checkpoints/20261119/idc-waymo-v1.0_examples_041839_episode=475.pth

echo "=== 后台启动，观察 60 秒是否推进 ==="
docker exec -d $C bash -lc "cd $CR && IDC_STATE_V2=1 $PY -u src/scripts/train/idc-train.py \
  --data-dir /data/data_json/training --num-worlds 2 --epochs 1 --device cuda \
  --planner-runtime /workspace/data/plant2_ckpt/best.pt \
  --planner-replan-every 5 --planner-k 1 \
  --load-model --model-path $CK --save-freq 1 > /tmp/train_bg.log 2>&1"

sleep 60
echo "--- 60s 后日志行数与末 12 行 ---"
docker exec $C bash -lc "wc -l /tmp/train_bg.log; tail -12 /tmp/train_bg.log"
echo "--- 进程 ---"
docker exec $C bash -lc "pgrep -af 'idc-train.py' | head -3"
