#!/bin/bash
set -u
C=idc_hist
CR=/gpudrive/baselines/IDC-Bridge
PY=/gpudrive/.venv/bin/python
CK=/workspace/data/checkpoints/20261119/idc-waymo-v1.0_examples_041839_episode=475.pth

echo "=== 打补丁 ==="
docker exec $C bash -lc "cd $CR && $PY tools/patch_reset_sched.py 2>&1 | tail -3"

echo "=== 训练冒烟：2 world × 1 epoch（重置调度）==="
docker exec $C bash -lc "cd $CR && IDC_STATE_V2=1 IDC_RESET_SCHEDULE=1 timeout 900 $PY -u src/scripts/train/idc-train.py \
  --data-dir /data/data_json/training --num-worlds 2 --epochs 1 --device cuda \
  --planner-runtime /workspace/data/plant2_ckpt/best.pt \
  --planner-replan-every 5 --planner-k 1 \
  --load-model --model-path $CK --save-freq 1 > /tmp/train_ok.log 2>&1; echo exit=\$?"

echo "--- TRACE / 步数 / loss ---"
docker exec $C bash -lc "grep -nE 'TRACE|回合|loss|Loss|update|保存' /tmp/train_ok.log | head -14"
echo "--- 末尾 ---"
docker exec $C bash -lc "tail -8 /tmp/train_ok.log"
