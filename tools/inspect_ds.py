"""打印 npz 数据集的字段/形状/基本合理性。"""
import sys

import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else '/workspace/data/tiny6/train.npz'
z = np.load(path, allow_pickle=True)
print('== %s  (%d 条) ==' % (path, len(z['ego'])))
for k in z.files:
    a = z[k]
    print('%-16s %-18s %s' % (k, str(a.shape), a.dtype))

if 'map_r' in z.files:
    lane = (z['map_r'][:, :, 4] > 0.5).sum(1)
    print('n_map_r: %s' % z['n_map_r'][:8])
    print('map_r lane 条数: %s' % lane[:8])
    print('map_r 是否达到上限 512 的样本数: %d/%d'
          % (int((z['n_map_r'] >= 512).sum()), len(z['n_map_r'])))
    mr = z['map_r'][0]
    print('map_r[0] 前 3 条: %s' % np.round(mr[:3], 2))
if 'agent_fut_head' in z.files:
    nz = (np.abs(z['agent_fut_head'][:, 0]) > 1e-6).sum(1)
    print('t=0 帧有朝向的周车数: %s' % nz[:8])
    print('amask: %s' % z['amask'].sum(1)[:8])
if 'expert_head40' in z.files:
    print('expert_head40[0][:5]: %s' % np.round(z['expert_head40'][0][:5], 3))
    print('ego_dim: %s' % np.round(z['ego_dim'][:3], 2))
