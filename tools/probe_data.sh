#!/bin/bash
set -u
docker exec idc_hist bash -lc '
cd /gpudrive/baselines/IDC-Bridge
/gpudrive/.venv/bin/python -c "
import sys; sys.path.insert(0,\"src\")
from utils.config import build_config
cfg = build_config(\"src/configs/eval.yaml\")
print(\"file_dir =\", repr(getattr(cfg,\"file_dir\",None)))
print(\"dataset_size =\", getattr(cfg,\"dataset_size\",None))
import os
d = getattr(cfg,\"file_dir\",\"\")
print(\"exists:\", os.path.isdir(d))
if os.path.isdir(d):
    fs = os.listdir(d)
    print(\"条目数:\", len(fs), fs[:5])
" 2>&1 | tail -6'
echo "=== 找 tfrecord/json 数据集目录 ==="
docker exec idc_hist bash -lc '
for d in /workspace/data /data /gpudrive/data; do
  echo "-- $d"; ls $d 2>/dev/null | head -6
done
echo "-- 含 tfrecord 的目录"
find /workspace/data -maxdepth 3 -name "*.tfrecord" 2>/dev/null | head -3
find /workspace/data -maxdepth 2 -name "tfrecord*" -o -maxdepth 2 -name "dataset*" 2>/dev/null | head -5
'
