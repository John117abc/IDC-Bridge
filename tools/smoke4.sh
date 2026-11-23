#!/bin/bash
set -u
echo "=== 数据目录确认 ==="
docker exec idc_hist bash -lc 'ls /data/data_json/training | head -2; echo "总数: $(ls /data/data_json/training | wc -l)"'

echo "=== 冒烟 ==="
docker exec idc_hist bash -lc '
cd /gpudrive/baselines/IDC-Bridge
timeout 900 /gpudrive/.venv/bin/python tools/collect_v2_pairs.py \
  --ckpt /workspace/data/plant2_ckpt/best.pt \
  --out /workspace/data/v2_smoke.npz \
  --worlds 2 --steps 12 --k 1 --data_root /data/data_json/training 2>&1 | tail -18
'
echo "=== 产物 ==="
docker exec idc_hist bash -lc '
/gpudrive/.venv/bin/python - <<PY
import numpy as np
z = np.load("/workspace/data/v2_smoke.npz")
print("keys", z.files)
for k in z.files: print("  ", k, z[k].shape)
s = z["state"]
print("state 范围 min %.2f max %.2f |x|max %.2f" % (s.min(), s.max(), abs(s).max()))
r = z["refpos"]
print("refpos 首末点示例:", np.round(r[0,0],2), np.round(r[0,-1],2))
PY'
