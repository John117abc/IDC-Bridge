#!/bin/bash
set -u
echo "=== idc-eval 的 __main__/argparse ==="
sed -n "$(grep -n '__main__' /gpudrive/baselines/IDC-Bridge/src/scripts/eval/idc-eval.py | cut -d: -f1),+40p" /gpudrive/baselines/IDC-Bridge/src/scripts/eval/idc-eval.py
echo
echo "=== config 里 planner_runtime 等字段默认值 ==="
grep -nE "planner_runtime|planner_replan_every|planner_k|data_dir|max_step|num_worlds" /gpudrive/baselines/IDC-Bridge/src/configs/eval.yaml | head -12
