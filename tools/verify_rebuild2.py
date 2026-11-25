"""重建完整性检查（精度修正）：通道平均格数"""
import glob
import os

import numpy as np

D = '/mnt/2T_HDD/plant2_vector/vector_ds_occ'
rows = []
for f in sorted(glob.glob(os.path.join(D, '*.npz'))):
    d = np.load(f)
    n = len(d['ego'])
    o = d['occ']
    rows.append((
        os.path.basename(f)[:-4], n,
        float(o[:, 0].sum(axis=(1, 2)).mean()),
        float(o[:, 1].sum(axis=(1, 2)).mean()),
        float(o[:, 2].sum(axis=(1, 2)).mean()),
    ))

print(f'{"scene":>34} {"n":>7} {"车格":>7} {"人格":>7} {"静态格":>7}')
for r in sorted(rows, key=lambda x: -x[1])[:10]:
    print(f'{r[0]:>34} {r[1]:>7} {r[2]:>7.1f} {r[3]:>7.2f} {r[4]:>7.2f}')

print('\n=== 关键场景 ===')
for r in rows:
    if any(k in r[0] for k in ('Construction', 'Pedestrian', 'CrossingBicycle')):
        print(f'{r[0]:>34} n={r[1]:>6} 车={r[2]:>6.1f} 人={r[3]:>6.2f} 静态={r[4]:>6.2f}')

tot = sum(r[1] for r in rows)
print(f'\n文件 {len(rows)}，总样本 {tot}')
print('静态通道全为 0 的场景数:',
      sum(1 for r in rows if r[4] < 0.01))
print('行人通道全为 0 的场景数:',
      sum(1 for r in rows if r[3] < 0.01))
