#!/bin/bash
set -u
C=idc_hist
R=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge
echo "=== __main__ 段 ==="
docker exec $C bash -lc "sed -n \"\$(grep -n '__main__' $R/src/scripts/eval/idc-eval.py | cut -d: -f1),+35p\" $R/src/scripts/eval/idc-eval.py" 2>&1 | tail -40
echo
echo "=== eval.yaml 关键字段 ==="
docker exec $C bash -lc "grep -nE 'planner|data_dir|max_step|num_worlds|epochs' $R/src/configs/eval.yaml" 2>&1 | head -15
