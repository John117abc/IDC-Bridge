"""调试：训练里的候选（未训练模型输出）为什么奖励恒 0。"""
import sys

import numpy as np
import torch

sys.path.insert(0, '/gpudrive/baselines/IDC-Bridge/src')
sys.path.insert(0, '/gpudrive/baselines/IDC-Bridge/tools')

from metrics.openloop_pdms_vec import geo_from_npz, score_batch  # noqa
from train_planner import Planner, densify  # noqa

z = np.load('/workspace/data/planner300/train.npz', allow_pickle=True)
torch.manual_seed(0)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

i = 0
m = Planner(K=6).to(dev)
with torch.no_grad():
    pred = m(torch.tensor(z['ego'][i:i + 1], dtype=torch.float32, device=dev),
             torch.tensor(z['agent'][i:i + 1], dtype=torch.float32, device=dev),
             torch.tensor(z['amask'][i:i + 1], dtype=torch.float32, device=dev),
             torch.tensor(z['map'][i:i + 1], dtype=torch.float32, device=dev),
             torch.tensor(z['mmask'][i:i + 1], dtype=torch.float32, device=dev))
print('pred 范围 %.2f ~ %.2f' % (float(pred.min()), float(pred.max())))
p = pred[0].cpu().numpy()
base = densify(p[0])
print('base 末点 %s' % np.round(base[-1], 2))
print('gt 末点 %s' % np.round(z['gt'][i][-1], 2))

g = geo_from_npz(z, i)
rng = np.random.default_rng(0)
ramp = (np.linspace(0, 1, 40) ** 2)[:, None]
cands = np.empty((6 * 4, 40, 2))
for k in range(6):
    b40 = densify(p[k])
    eps = rng.normal(0, 1.0, (4, 2))
    cands[k * 4:(k + 1) * 4] = b40[None] + 0.5 * ramp[None] * eps[:, None, :]
o = score_batch(g, cands)
print('PDMS %s' % np.round(o['driving_score'], 2))
print('nc %s' % o['nc'])
print('dac %s' % o['dac'])
print('ddc %s' % o['ddc'])
print('ep %s' % np.round(o['ep'], 3))
print('ttc %s' % np.round(o['ttc'], 3))
print('comfort %s' % np.round(o['comfort'], 3))
print('lk %s' % np.round(o['lk'], 3))
print('coll 步数 %s' % o['coll'].sum(1))
print('off 步数 %s' % o['off'].sum(1))
print('lat p50 %.2f hm_lat p50 %.2f  edge 数 %d lane 数 %d'
      % (np.median(o['lat']), np.median(g.hm['lat']), len(g.ea), len(g.la)))
print('cand 坐标范围 %s ~ %s' % (np.round(cands.reshape(-1, 2).min(0), 2),
                                 np.round(cands.reshape(-1, 2).max(0), 2)))
dd = np.linalg.norm(cands.reshape(-1, 2)[:, None] - np.asarray([[0., 0.]])[None], axis=2)
print('离原点最远 %.1f m' % dd.max())
