#!/bin/bash
set -u
C=idc_hist
CR=/gpudrive/baselines/IDC-Bridge
CK=/workspace/data/checkpoints/20261119/idc-waymo-v1.0_examples_041839_episode=475.pth
PY=/gpudrive/.venv/bin/python
TMP=$CR/src/configs/_tmp_eval_rt.yaml

echo "=== 写临时 yaml（放 configs 内，保证 base.yaml 可解析）==="
docker exec $C bash -lc "cp $CR/src/configs/eval.yaml $TMP && \
  printf '\nplanner_runtime: /workspace/data/plant2_ckpt/best.pt\nplanner_replan_every: 5\nplanner_k: 1\nplanner_mode: planner\n' >> $TMP && tail -4 $TMP"

echo
echo "=== 带规划器跑 2 world × 1 epoch ==="
docker exec $C bash -lc "cd $CR && timeout 900 $PY src/scripts/eval/idc-eval.py \
  --config $TMP --data-dir /data/data_json/training --model-path $CK \
  --num-worlds 2 --epochs 1 --gif-enabled false > /tmp/eval_rt2.log 2>&1; echo exit=\$?"

echo "--- RT / 特征日志 ---"
docker exec $C bash -lc "grep -E 'RT\]|ScenePack|Traceback|PDMS-rollout' /tmp/eval_rt2.log | head -14"
echo "--- 结尾 ---"
docker exec $C bash -lc "tail -5 /tmp/eval_rt2.log"
