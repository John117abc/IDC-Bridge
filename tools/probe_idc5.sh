#!/bin/bash
R=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge
F=$R/src/env/idc_state_builder.py
echo "=== 文件结构 ==="
grep -nE "^class |^    def |^def " "$F" | head -30
echo
echo "=== 行数 ==="
wc -l "$F"
echo
echo "=== checkpoint keys（weights_only=False）==="
docker exec -i idc_hist /gpudrive/.venv/bin/python -c "
import torch
ck = torch.load('/workspace/data/checkpoints/20261119/idc-waymo-v1.0_examples_041839_episode=475.pth', map_location='cpu', weights_only=False)
print('TOP', list(ck.keys()))
for k, v in ck.items():
    if hasattr(v, 'keys'):
        print(k, '->', len(v))
        for kk in list(v.keys())[:4]:
            try: print('   ', kk, tuple(v[kk].shape))
            except Exception: pass
" 2>&1 | head -14
