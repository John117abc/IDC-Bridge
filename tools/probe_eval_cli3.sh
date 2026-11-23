#!/bin/bash
set -u
C=idc_hist
CR=/gpudrive/baselines/IDC-Bridge
echo "=== __main__ 段 ==="
docker exec $C bash -lc "n=\$(grep -n '__main__' $CR/src/scripts/eval/idc-eval.py | cut -d: -f1); sed -n \"\${n},+30p\" $CR/src/scripts/eval/idc-eval.py" 2>&1 | tail -34
echo
echo "=== eval.yaml 关键字段 ==="
docker exec $C bash -lc "grep -nE 'planner|data_dir|max_step|num_worlds|epochs' $CR/src/configs/eval.yaml" 2>&1 | head -15
