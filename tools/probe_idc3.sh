#!/bin/bash
R=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge
echo "=== 启动容器 ==="
docker start idc_hist >/dev/null 2>&1
sleep 8
docker exec idc_hist nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>&1 | head -2

echo "=== checkpoint 维度 ==="
docker exec idc_hist /gpudrive/.venv/bin/python - <<'EOF' 2>&1 | tail -20
import torch
p = '/workspace/data/checkpoints/20261119/idc-waymo-v1.0_examples_041839_episode=475.pth'
ck = torch.load(p, map_location='cpu')
print('top keys:', list(ck.keys()))
sd = ck.get('model', ck)
if isinstance(sd, dict) and 'actor' in sd:
    print('has actor; keys:', list(sd.keys()))
    sd = sd['actor']
ks = list(sd.keys())
for k in ks[:12]:
    try: print(k, tuple(sd[k].shape))
    except Exception: pass
EOF

echo "=== per_buffer.py 里状态组装 ==="
grep -nE "def |ref_start|dphi|dp|curva|road|spd" "$R/src/buffer/per_buffer.py" | head -30
