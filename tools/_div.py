import numpy as np
z = np.load('/mnt/2T_HDD/plant2_vector/dbg.npz')
m = z['modes']
K = m.shape[1]
d = np.linalg.norm(m[:, :, -1, None, :] - m[:, None, :, -1, :], axis=-1)
kk = ~np.eye(K, dtype=bool)
mp = np.where(kk[None], d, np.inf).min(axis=(1, 2))
print('frames', len(m))
print('min-pair endpoint dist: mean %.2f median %.2f' % (mp.mean(), np.median(mp)))
print('frac frames all pairs >2m: %.2f' % float((mp > 2).mean()))
print('selected unique count:', len(np.unique(z['k'])) if 'k' in z.files else 'n/a')
