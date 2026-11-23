#!/bin/bash
set -u
C=idc_hist
CR=/gpudrive/baselines/IDC-Bridge
CK=/workspace/data/checkpoints/20261119/idc-waymo-v1.0_examples_041839_episode=475.pth
PY=/gpudrive/.venv/bin/python

echo "=== 官方 eval + 在线规划器（2 world × 1 epoch）==="
docker exec $C bash -lc "cd $CR && timeout 900 $PY src/scripts/eval/idc-eval.py \
  --data-dir /data/data_json/training --model-path $CK \
  --planner-runtime /workspace/data/plant2_ckpt/best.pt \
  --planner-replan-every 5 --planner-k 1 \
  --num-worlds 2 --epochs 1 --gif-enabled false \
  --dump-feat /workspace/data/feat_online.npz > /tmp/eval_rt3.log 2>&1; echo exit=\$?"

echo "--- RT 特征日志（关键：map n=?）---"
docker exec $C bash -lc "grep -E 'RT\]|ScenePack' /tmp/eval_rt3.log | head -12"
echo "--- PDMS ---"
docker exec $C bash -lc "grep -E 'PDMS-rollout' /tmp/eval_rt3.log | head -4"
echo "--- 错误（若有）---"
docker exec $C bash -lc "grep -E 'Traceback|Error' /tmp/eval_rt3.log | head -5"
