#!/bin/bash
R=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge
echo "=== A) checkpoint 元数据 ==="
docker exec -i idc_hist /gpudrive/.venv/bin/python -c "
import torch
ck = torch.load('/workspace/data/checkpoints/20261119/idc-waymo-v1.0_examples_041839_episode=475.pth', map_location='cpu', weights_only=False)
for k in ('state_dim','state_mode','ref_mode'):
    print(k,'=',ck.get(k))
try:
    print('config:', {kk: vv for kk, vv in ck['config'].__dict__.items() if not kk.startswith('_')})
except Exception as e:
    print('config raw:', str(ck.get('config'))[:400])
" 2>&1 | head -12

echo
echo "=== B) _calc_ref_error 全文 ==="
sed -n '1195,1215p' "$R/src/env/idc_state_builder.py"

echo
echo "=== C) 组装尾部（960-1050）==="
sed -n '960,1050p' "$R/src/env/idc_state_builder.py"
