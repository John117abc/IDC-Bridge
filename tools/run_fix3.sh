#!/bin/bash
set -u
C=idc_hist
CR=/gpudrive/baselines/IDC-Bridge
PY=/gpudrive/.venv/bin/python
CK=/workspace/data/checkpoints/20261119/idc-waymo-v1.0_examples_041839_episode=475.pth
docker exec $C bash -lc "cd $CR && timeout 900 $PY src/scripts/eval/idc-eval.py \
  --data-dir /data/data_json/training --model-path $CK \
  --planner-runtime /workspace/data/plant2_ckpt/best.pt \
  --planner-replan-every 5 --planner-k 1 \
  --num-worlds 2 --epochs 1 --gif-enabled false > /tmp/eval_fix3.log 2>&1; echo exit=\$?"
echo "--- align / feat ---"
docker exec $C bash -lc "grep -E 'align|RT..\[feat\]' /tmp/eval_fix3.log | head -8"
echo "--- PDMS ---"
docker exec $C bash -lc "grep -E 'PDMS-rollout' /tmp/eval_fix3.log | head -3"
