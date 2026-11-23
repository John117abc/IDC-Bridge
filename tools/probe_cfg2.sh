#!/bin/bash
set -u
echo "=== build_config(eval.yaml) 的字段 ==="
docker exec idc_hist bash -lc '
cd /gpudrive/baselines/IDC-Bridge
/gpudrive/.venv/bin/python -c "
import os, sys
sys.path.insert(0, \"src\")
from utils.config import build_config
cfg = build_config(\"src/configs/eval.yaml\")
ks = sorted(vars(cfg).keys())
print(\"字段数:\", len(ks))
print([k for k in ks if \"data\" in k or \"dir\" in k or \"root\" in k or \"path\" in k])
print(ks[:40])
" 2>&1 | tail -8'
echo "=== 数据集目录候选 ==="
docker exec idc_hist bash -lc 'ls /workspace/data | head -12'
