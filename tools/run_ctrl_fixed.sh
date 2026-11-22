#!/bin/bash
# 控制实验：用 env 自带的候选安装（固定横向偏移）看 RC 会不会也掉
set -u
cd /gpudrive/baselines/IDC-Bridge || exit 1
P=/gpudrive/.venv/bin/python
CKPT=/workspace/data/checkpoints/20261119/idc-waymo-v1.0_examples_041839_episode=475.pth
L=/workspace/data/logs
D=/data/data_json/training
W=${W:-12}

for K in 3 5; do
  $P src/scripts/eval/idc-eval.py --data-dir $D --model-path "$CKPT" \
      --num-worlds $W --seed 42 --cand-fixed $K \
      --gif-enabled false > "$L/ctrl_fixed_$K.log" 2>&1
  echo "--- cand-fixed K=$K（env 自带候选）---"
  grep -E "Driving Score|Route Completion|NC |固定候选" "$L/ctrl_fixed_$K.log" | tail -4
done

echo "--- 基线（无候选安装）---"
grep -E "Driving Score|Route Completion|NC " "$L/joint_base.log" | tail -3
