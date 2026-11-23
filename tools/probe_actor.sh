#!/bin/bash
set -u
CR=/gpudrive/baselines/IDC-Bridge
echo "=== TransformerActor 定义 ==="
docker exec idc_hist bash -lc "grep -rn 'class TransformerActor' -A 40 $CR/src/agents/actor.py 2>/dev/null | head -45 || grep -rn 'class TransformerActor' -A 40 $CR/src/agents/idc_agent.py | head -45"
echo
echo "=== RunningNorm ==="
docker exec idc_hist bash -lc "sed -n '1,45p' $CR/src/utils/running_norm.py"
