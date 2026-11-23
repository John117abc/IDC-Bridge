#!/bin/bash
set -u
CR=/gpudrive/baselines/IDC-Bridge
echo "=== TransformerActor / input_proj 位置 ==="
docker exec idc_hist bash -lc "grep -rn 'class TransformerActor' $CR/src | head -5"
docker exec idc_hist bash -lc "grep -rn 'input_proj' $CR/src | head -8"
echo
echo "=== RunningNorm 其余部分（stats/__call__/load）==="
docker exec idc_hist bash -lc "sed -n '45,120p' $CR/src/utils/running_norm.py"
