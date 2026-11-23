#!/bin/bash
set -u
C=idc_hist
CR=/gpudrive/baselines/IDC-Bridge
PY=/gpudrive/.venv/bin/python
CK=/workspace/data/checkpoints/20261119/idc-waymo-v1.0_examples_041839_episode=475.pth

echo "=== 3 epoch × 4 world（v2 + 规划器）==="
docker exec $C bash -lc "cd $CR && IDC_STATE_V2=1 IDC_RESET_SCHEDULE=1 timeout 1800 $PY -u src/scripts/train/idc-train.py \
  --data-dir /data/data_json/training --num-worlds 4 --epochs 3 --device cuda \
  --planner-runtime /workspace/data/plant2_ckpt/best.pt \
  --planner-replan-every 5 --planner-k 1 \
  --load-model --model-path $CK --save-freq 1 > /tmp/train3.log 2>&1; echo exit=\$?"

echo "--- 规划器是否激活 ---"
docker exec $C bash -lc "grep -cE 'RT\]' /tmp/train3.log; grep -E 'RT\]\[align|RT\]\[feat' /tmp/train3.log | head -3"
echo "--- 每轮 PDMS / loss ---"
docker exec $C bash -lc "grep -E 'PDMS\]|Critic Loss|Actor Loss|DIAG-ref' /tmp/train3.log | tail -10"
echo "--- 保存 ---"
docker exec $C bash -lc "grep -E '保存模型' /tmp/train3.log | tail -3"
