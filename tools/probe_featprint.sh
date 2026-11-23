#!/bin/bash
set -u
C=idc_hist
CR=/gpudrive/baselines/IDC-Bridge
echo "=== [RT][feat] 打印处的上下文 ==="
docker exec $C bash -lc "grep -n 'RT\]\[feat\]' $CR/src/env/planner_runtime.py"
echo "--- 上下文 ---"
docker exec $C bash -lc "n=\$(grep -n 'RT\]\[feat\]' $CR/src/env/planner_runtime.py | head -1 | cut -d: -f1); sed -n \"\$((n-30)),\$((n+8))p\" $CR/src/env/planner_runtime.py"
