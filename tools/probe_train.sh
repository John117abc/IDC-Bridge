#!/bin/bash
set -u
C=idc_hist
CR=/gpudrive/baselines/IDC-Bridge
echo "=== idc-train.py 的 CLI 参数 ==="
docker exec $C bash -lc "grep -nE \"add_argument\" $CR/src/scripts/train/idc-train.py | head -30"
echo
echo "=== 是否接线 PlannerRuntime ==="
docker exec $C bash -lc "grep -nE 'planner_runtime|PlannerRuntime|install_external' $CR/src/scripts/train/idc-train.py | head -10"
echo
echo "=== 训练主循环关键调用 ==="
docker exec $C bash -lc "grep -nE 'get_idc_observations|select_action|store_transition|env.step|reward' $CR/src/scripts/train/idc-train.py | head -14"
