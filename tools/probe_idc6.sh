#!/bin/bash
R=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge

echo "=== 1) checkpoint 元数据（state_dim / state_mode / ref_mode）==="
docker exec -i idc_hist /gpudrive/.venv/bin/python -c "
import torch
ck = torch.load('/workspace/data/checkpoints/20261119/idc-waymo-v1.0_examples_041839_episode=475.pth', map_location='cpu', weights_only=False)
for k in ('state_dim','state_mode','ref_mode','globe_eps','rho','gep_iteration'):
    print(k, '=', ck.get(k))
print('config keys:', list(ck['config'].__dict__.keys())[:25] if hasattr(ck.get('config'),'__dict__') else ck.get('config'))
print('history:', ck.get('history'))
" 2>&1 | head -14

echo
echo "=== 2) _calc_ref_error（1195-1230）==="
sed -n '1195,1230p' "$R/src/env/idc_state_builder.py"

echo
echo "=== 3) get_idc_observations_batch 的状态拼接（847-960）==="
sed -n '847,960p' "$R/src/env/idc_state_builder.py"
