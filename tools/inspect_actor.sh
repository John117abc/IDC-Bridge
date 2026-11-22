#!/bin/bash
# ?????? checkpoint ??actor ?????? / ???????????????cd /gpudrive/baselines/IDC-Bridge || exit 1
/gpudrive/.venv/bin/python - <<'PY'
import torch
p = '/workspace/data/checkpoints/20261119/idc-waymo-v1.0_examples_041839_episode=475.pth'
ck = torch.load(p, map_location="cpu", weights_only=False)
print('?????', list(ck.keys())[:8])
sd = ck.get('model', ck.get('actor', ck))
if isinstance(sd, dict):
    for k, v in sd.items():
        if hasattr(v, 'shape') and ('head' in k or 'out' in k or 'cand' in k):
            print('%-50s %s' % (k, tuple(v.shape)))
PY

