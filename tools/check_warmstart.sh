#!/bin/bash
set -u
C=idc_hist
CR=/gpudrive/baselines/IDC-Bridge
PY=/gpudrive/.venv/bin/python
OLD=/workspace/data/checkpoints/20261119/idc-waymo-v1.0_examples_041839_episode=475.pth

echo "=== 旧权重 + 84 维接口（验证热启动等价性）==="
docker exec $C bash -lc "cd $CR && IDC_STATE_V2=1 timeout 900 $PY src/scripts/eval/idc-eval.py \
  --data-dir /data/data_json/training --model-path $OLD \
  --num-worlds 8 --epochs 1 --seed 42 --gif-enabled false > /tmp/ev_v84old.log 2>&1; echo exit=\$?"
docker exec $C bash -lc "grep -oE 'predicted_score=[0-9.]+' /tmp/ev_v84old.log | sed 's/.*=//' | tr '\n' ' '; echo"
docker exec $C bash -lc "grep -oE 'predicted_score=[0-9.]+' /tmp/ev_v84old.log | sed 's/.*=//' | awk '{s+=\$1;n++} END{printf \"均值 %.1f (n=%d)\\n\", s/n, n}'"
