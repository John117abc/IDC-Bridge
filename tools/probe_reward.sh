#!/bin/bash
set -u
C=idc_hist
CR=/gpudrive/baselines/IDC-Bridge
echo "=== reward 里用到的参考来源 ==="
docker exec $C bash -lc "grep -nE 'expert_pos|candidate_paths|ref_path|pos_err|lat_err|progress' $CR/src/env/world_manager.py | head -25"
echo
echo "=== agent 里算 reward 的位置 ==="
docker exec $C bash -lc "grep -nE 'def .*reward|expert_pos|candidate_paths|pos_err|lat_err' $CR/src/agents/idc_agent.py | head -25"
