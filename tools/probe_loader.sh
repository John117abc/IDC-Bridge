#!/bin/bash
set -u
docker exec idc_hist bash -lc 'sed -n "20,75p" /gpudrive/gpudrive/env/dataset.py'
echo "=== training 目录里到底有什么 ==="
docker exec idc_hist bash -lc 'ls /data/data/training | head -3; echo "总数: $(ls /data/data/training | wc -l)"'
echo "=== testing ==="
docker exec idc_hist bash -lc 'ls /data/data/testing | head -3; echo "总数: $(ls /data/data/testing | wc -l)"'
echo "=== 我们之前 eval 用的数据目录（从历史日志找） ==="
docker exec idc_hist bash -lc 'grep -rhoE "Data root: [^ ]+" /workspace/data/logs 2>/dev/null | sort -u | head -5'
