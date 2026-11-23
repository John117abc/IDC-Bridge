#!/bin/bash
set -u
echo "=== /data/data/training 内容 ==="
docker exec idc_hist bash -lc 'ls /data/data/training | head -3; echo "文件数: $(ls /data/data/training | wc -l)"'

echo "=== 冒烟（root=/data/data/training）==="
docker exec idc_hist bash -lc '
cd /gpudrive/baselines/IDC-Bridge
timeout 900 /gpudrive/.venv/bin/python tools/collect_v2_pairs.py \
  --ckpt /workspace/data/plant2_ckpt/best.pt \
  --out /workspace/data/v2_smoke.npz \
  --worlds 2 --steps 12 --k 1 --data_root /data/data/training 2>&1 | tail -20
'
echo "=== 产物 ==="
docker exec idc_hist bash -lc '
/gpudrive/.venv/bin/python -c "
import numpy as np
z = np.load(\"/workspace/data/v2_smoke.npz\")
print(\"keys\", z.files)
for k in z.files: print(\" \", k, z[k].shape)
s = z[\"state\"]
print(\"state 范围: min %.2f max %.2f  |x|max %.2f\" % (s.min(), s.max(), abs(s).max()))
" 2>&1 | tail -12'
