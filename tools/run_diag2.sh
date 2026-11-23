#!/bin/bash
set -u
C=idc_hist
CR=/gpudrive/baselines/IDC-Bridge
PY=/gpudrive/.venv/bin/python
CK=/workspace/data/checkpoints/20261119/idc-waymo-v1.0_examples_041839_episode=475.pth

docker exec $C bash -lc "cd $CR && $PY tools/patch_diag_pack.py 2>&1 | tail -3"
echo "=== 跑 1 world 1 epoch ==="
docker exec $C bash -lc "cd $CR && timeout 600 $PY src/scripts/eval/idc-eval.py \
  --data-dir /data/data_json/training --model-path $CK \
  --planner-runtime /workspace/data/plant2_ckpt/best.pt \
  --planner-replan-every 5 --planner-k 1 \
  --num-worlds 1 --epochs 1 --gif-enabled false > /tmp/eval_diag.log 2>&1; echo exit=\$?"
echo "--- pack 诊断 ---"
docker exec $C bash -lc "grep -E 'RT\]\[pack\]|shift 失败|ScenePack' /tmp/eval_diag.log | head -5"
echo "--- feat ---"
docker exec $C bash -lc "grep -E 'RT\]\[feat\]' /tmp/eval_diag.log | head -3"
