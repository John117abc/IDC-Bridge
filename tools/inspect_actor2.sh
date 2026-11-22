#!/bin/bash
cd /gpudrive/baselines/IDC-Bridge || exit 1
/gpudrive/.venv/bin/python - <<'PY'
import torch
p = '/workspace/data/checkpoints/20261119/idc-waymo-v1.0_examples_041839_episode=475.pth'
ck = torch.load(p, map_location='cpu', weights_only=False)
print('global_step', ck.get('global_step'), 'state_dim', ck.get('state_dim'))
cfg = ck.get('config')
if cfg is not None:
    for k in dir(cfg):
        if k.startswith('_'):
            continue
        v = getattr(cfg, k)
        if isinstance(v, (int, float, str, bool)):
            print('  cfg.%s = %s' % (k, v))
sd = ck['model_state_dict']
ks = list(sd.keys())
print('参数张量数', len(ks))
for k in ks[-14:]:
    print('%-52s %s' % (k, tuple(sd[k].shape)))
PY
