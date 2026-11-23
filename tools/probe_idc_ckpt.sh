#!/bin/bash
PY=/mnt/2T_HDD/conda_envs/bench2drive_b2d/bin/python
R=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge
CKPT=/workspace/data/checkpoints/20261119/idc-waymo-v1.0_examples_041839_episode=475.pth

echo "=== 1) checkpoint 的 actor 输入维度 ==="
docker exec idc_hist "$PY" - <<'EOF' 2>/dev/null || echo "容器不可用，改用宿主机"
import torch
ck = torch.load('/workspace/data/checkpoints/20261119/idc-waymo-v1.0_examples_041839_episode=475.pth',
                map_location='cpu')
sd = ck.get('model', ck.get('actor', ck))
keys = [k for k in sd if 'actor' in k or 'policy' in k or 'pi' in k][:8]
for k in keys:
    print(k, tuple(sd[k].shape))
EOF

echo "=== 2) 状态组装（world_manager.py 143-190）==="
sed -n '143,190p' "$R/src/env/world_manager.py"
