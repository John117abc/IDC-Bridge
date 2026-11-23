#!/bin/bash
set -u
CR=/gpudrive/baselines/IDC-Bridge
echo "=== _project_index_torch (1275-1305) ==="
docker exec idc_hist bash -lc "sed -n '1275,1305p' $CR/src/env/idc_state_builder.py"
echo
echo "=== get_ref_states_batch (1315-1355) ==="
docker exec idc_hist bash -lc "sed -n '1315,1355p' $CR/src/env/idc_state_builder.py"
