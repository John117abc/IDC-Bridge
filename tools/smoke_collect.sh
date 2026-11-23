#!/bin/bash
set -u
# 冒烟：在容器内用规划器产参考，采集 v2 状态对
HOST_CK=/mnt/2T_HDD/WaymoDatabas/plant2_ckpt/best.pt
[ -f "$HOST_CK" ] || { echo "缺少规划器权重: $HOST_CK"; exit 1; }

echo "=== 冒烟：2 world × 12 步 ==="
docker exec idc_hist bash -lc '
cd /gpudrive/baselines/IDC-Bridge
timeout 900 /gpudrive/.venv/bin/python tools/collect_v2_pairs.py \
  --ckpt /workspace/data/plant2_ckpt/best.pt \
  --out /workspace/data/v2_smoke.npz \
  --worlds 2 --steps 12 --k 1 2>&1 | tail -30
'
echo "=== 产物 ==="
docker exec idc_hist bash -lc '
/workspace/../gpudrive/.venv/bin/python -c "
import numpy as np
z = np.load(\"/workspace/data/v2_smoke.npz\")
print(\"keys\", z.files)
for k in z.files: print(k, z[k].shape)
" 2>&1 | tail -10'
