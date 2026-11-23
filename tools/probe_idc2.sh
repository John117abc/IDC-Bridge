#!/bin/bash
R=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge
CKPT=/workspace/data/checkpoints/20261119/idc-waymo-v1.0_examples_041839_episode=475.pth

echo "===== 1) checkpoint keys / actor 输入维度 ====="
docker exec idc_hist /gpudrive/.venv/bin/python - <<'EOF' 2>&1 | tail -25
import torch
ck = torch.load('/workspace/data/checkpoints/20261119/idc-waymo-v1.0_examples_041839_episode=475.pth',
                map_location='cpu')
print('top keys:', list(ck.keys())[:10])
sd = ck.get('model', ck)
if isinstance(sd, dict) and 'actor' in sd:
    sd = sd['actor']
ks = list(sd.keys())
print('n params:', len(ks))
for k in ks[:14]:
    try:
        print(k, tuple(sd[k].shape))
    except Exception:
        pass
EOF

echo
echo "===== 2) ref_error 18 维的组装位置 ====="
grep -nE "ref_error|dphi|dp\b|lat_err|curva_ahead|spd_ahead|road_dist" "$R/src/env/world_manager.py" | head -25
echo "--- candidate_sampler 里的辅助函数 ---"
grep -nE "^def |^    def " "$R/src/env/candidate_sampler.py" | head -25
