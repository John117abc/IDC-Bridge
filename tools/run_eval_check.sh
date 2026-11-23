#!/bin/bash
set -u
C=idc_hist
CR=/gpudrive/baselines/IDC-Bridge
CK=/workspace/data/checkpoints/20261119/idc-waymo-v1.0_examples_041839_episode=475.pth
PY=/gpudrive/.venv/bin/python

echo "=== A) 官方 eval 冒烟（不带规划器）==="
docker exec $C bash -lc "cd $CR && timeout 600 $PY src/scripts/eval/idc-eval.py \
  --data-dir /data/data_json/training --model-path $CK \
  --num-worlds 2 --epochs 1 --gif-enabled false > /tmp/eval_plain.log 2>&1; echo exit=\$?"
docker exec $C bash -lc "grep -E 'Data root|RT\]|PDMS|完成|Error|Traceback' /tmp/eval_plain.log | head -12"

echo
echo "=== B) 带规划器（临时 yaml 加 planner_runtime）==="
docker exec $C bash -lc "cp $CR/src/configs/eval.yaml /tmp/eval_rt.yaml && \
  printf '\nplanner_runtime: %s\nplanner_replan_every: 5\nplanner_k: 1\nplanner_mode: planner\n' $CR/../plant2_ckpt_placeholder > /dev/null 2>&1; \
  printf '\nplanner_runtime: /workspace/data/plant2_ckpt/best.pt\nplanner_replan_every: 5\nplanner_k: 1\nplanner_mode: planner\n' >> /tmp/eval_rt.yaml; \
  tail -5 /tmp/eval_rt.yaml"
docker exec $C bash -lc "cd $CR && timeout 600 $PY src/scripts/eval/idc-eval.py \
  --config /tmp/eval_rt.yaml \
  --data-dir /data/data_json/training --model-path $CK \
  --num-worlds 2 --epochs 1 --gif-enabled false > /tmp/eval_rt.log 2>&1; echo exit=\$?"
echo "--- 关键行 ---"
docker exec $C bash -lc "grep -E 'RT\]|Data root|Traceback|Error' /tmp/eval_rt.log | head -12"
