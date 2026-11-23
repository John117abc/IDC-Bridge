#!/bin/bash
set -u
C=idc_hist
CR=/gpudrive/baselines/IDC-Bridge
PY=/gpudrive/.venv/bin/python
CK=/workspace/data/checkpoints/20261119/idc-waymo-v1.0_examples_041839_episode=475.pth

echo "=== v2 冒烟（IDC_STATE_V2=1, 1 world）==="
docker exec $C bash -lc "cd $CR && IDC_STATE_V2=1 timeout 600 $PY src/scripts/eval/idc-eval.py \
  --data-dir /data/data_json/training --model-path $CK \
  --planner-runtime /workspace/data/plant2_ckpt/best.pt \
  --planner-replan-every 5 --planner-k 1 \
  --num-worlds 1 --epochs 1 --gif-enabled false > /tmp/eval_v2.log 2>&1; echo exit=\$?"

echo "--- 关键行 ---"
docker exec $C bash -lc "grep -E 'dim=|state_dim|维度|Traceback|PDMS-rollout|feat|align' /tmp/eval_v2.log | head -12"
echo "--- 错误尾部（若有）---"
docker exec $C bash -lc "grep -A6 'Traceback' /tmp/eval_v2.log | head -14"
