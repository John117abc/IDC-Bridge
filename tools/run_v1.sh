#!/bin/bash
set -u
C=idc_hist
CR=/gpudrive/baselines/IDC-Bridge
PY=/gpudrive/.venv/bin/python
CK=/workspace/data/checkpoints/20261119/idc-waymo-v1.0_examples_041839_episode=475.pth

echo "=== 跑 6 world，dump 规划器输出 ==="
docker exec $C bash -lc "cd $CR && timeout 1500 $PY src/scripts/eval/idc-eval.py \
  --data-dir /data/data_json/training --model-path $CK \
  --planner-runtime /workspace/data/plant2_ckpt/best.pt \
  --planner-replan-every 5 --planner-k 1 \
  --planner-dump /workspace/data/plan_dump.npz \
  --num-worlds 6 --epochs 1 --gif-enabled false > /tmp/eval_v1.log 2>&1; echo exit=\$?"

echo "--- PDMS ---"
docker exec $C bash -lc "grep -E 'PDMS-rollout' /tmp/eval_v1.log | head -8"
echo "--- dump 文件 ---"
docker exec $C bash -lc "ls -la /workspace/data/plan_dump.npz 2>/dev/null || echo '无 dump'"
