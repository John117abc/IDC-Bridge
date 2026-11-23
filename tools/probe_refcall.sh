#!/bin/bash
set -u
CR=/gpudrive/baselines/IDC-Bridge
echo "=== get_ref_states_batch 调用点 ==="
docker exec idc_hist bash -lc "grep -n 'get_ref_states_batch' $CR/src/agents/idc_agent.py"
echo "=== 调用上下文（update_critic）==="
docker exec idc_hist bash -lc "n=\$(grep -n 'get_ref_states_batch' $CR/src/agents/idc_agent.py | head -1 | cut -d: -f1); sed -n \"\$((n-18)),\$((n+10))p\" $CR/src/agents/idc_agent.py"
