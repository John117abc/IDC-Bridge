#!/bin/bash
set -u
C=idc_hist
CR=/gpudrive/baselines/IDC-Bridge
PY=/gpudrive/.venv/bin/python
OLD=/workspace/data/checkpoints/20261119/idc-waymo-v1.0_examples_041839_episode=475.pth
NEW=$(docker exec $C bash -lc "ls -t /workspace/data/checkpoints/20261122/*.pth | head -1" | tr -d '\r')
echo "新权重: $NEW"

echo "=== A) 旧模型（66 维）==="
docker exec $C bash -lc "cd $CR && timeout 900 $PY src/scripts/eval/idc-eval.py \
  --data-dir /data/data_json/training --model-path $OLD \
  --num-worlds 6 --epochs 1 --gif-enabled false > /tmp/eval_old66.log 2>&1; echo exit=\$?"
docker exec $C bash -lc "grep -oE 'predicted_score=[0-9.]+' /tmp/eval_old66.log | tr '\n' ' '; echo"

echo "=== B) 新模型（84 维，IDC_STATE_V2=1）==="
docker exec $C bash -lc "cd $CR && IDC_STATE_V2=1 timeout 900 $PY src/scripts/eval/idc-eval.py \
  --data-dir /data/data_json/training --model-path $NEW \
  --num-worlds 6 --epochs 1 --gif-enabled false > /tmp/eval_new84.log 2>&1; echo exit=\$?"
docker exec $C bash -lc "grep -oE 'predicted_score=[0-9.]+' /tmp/eval_new84.log | tr '\n' ' '; echo"
