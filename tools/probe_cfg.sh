#!/bin/bash
set -u
echo "=== 可用 config yaml ==="
docker exec idc_hist bash -lc 'ls /gpudrive/baselines/IDC-Bridge/src/configs/*.yaml 2>/dev/null | head -10'
echo "=== 其中 data_dir 指向 ==="
docker exec idc_hist bash -lc 'grep -h "data_dir" /gpudrive/baselines/IDC-Bridge/src/configs/*.yaml 2>/dev/null | sort -u | head -5'
