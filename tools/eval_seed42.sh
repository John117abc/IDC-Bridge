#!/bin/bash
set -u
C=idc_hist
CR=/gpudrive/baselines/IDC-Bridge
PY=/gpudrive/.venv/bin/python
OLD=/workspace/data/checkpoints/20261119/idc-waymo-v1.0_examples_041839_episode=475.pth
NEW=/workspace/data/checkpoints/20261122/idc-waymo-v1.0_examples_173804_episode=505.pth

run() {
  local tag=$1; local st=$2; local ck=$3
  docker exec $C bash -lc "cd $CR && IDC_STATE_V2=$st timeout 900 $PY src/scripts/eval/idc-eval.py \
    --data-dir /data/data_json/training --model-path $ck \
    --num-worlds 8 --epochs 1 --seed 42 --gif-enabled false > /tmp/ev_$tag.log 2>&1; echo exit=\$?"
  echo "--- $tag ---"
  docker exec $C bash -lc "grep -oE 'predicted_score=[0-9.]+' /tmp/ev_$tag.log | sed 's/predicted_score=//' | tr '\n' ' '; echo"
  docker exec $C bash -lc "grep -oE 'predicted_score=[0-9.]+' /tmp/ev_$tag.log | sed 's/.*=//' | awk '{s+=\$1;n++} END{printf \"均值 %.1f (n=%d)\\n\", s/n, n}'"
}

echo "=== 固定 seed=42, 8 worlds ==="
echo "=== 旧模型 66 维 ==="
run old 0 $OLD
echo "=== 新模型 84 维（训练 30 ep）==="
run new 1 $NEW
