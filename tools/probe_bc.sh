#!/bin/bash
set -u
C=idc_hist
CR=/gpudrive/baselines/IDC-Bridge
docker exec $C bash -lc "grep -rn 'bc_weight' $CR/src/agents/ $CR/src/scripts/train/ 2>/dev/null | head -12"
echo "--- expert_steer 使用处 ---"
docker exec $C bash -lc "grep -rn 'expert_steer' $CR/src/ 2>/dev/null | head -8"
