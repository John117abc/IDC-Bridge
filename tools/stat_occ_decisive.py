"""统计：数据集中有多少帧"必须看占据才能预测对"

指标（逐帧）：
  blocked  = 自车道走廊（x∈[1,16] m，|y|≤1.15 m）内有静态占据
  shift    = GT 轨迹最大横向偏移 > 0.6 m
  stop     = 自车速度 < 0.5 m/s
  decisive = blocked ∧ (shift ∨ stop)   ← 专家的行为对占据做出了反应
  gt_hit   = GT 轨迹自身与静态占据相交（专家撞了？应≈0）
"""
import glob
import os

import numpy as np

RES, GX0, GY0 = 0.5, -8.0, -16.0


def corridor_mask():
    """x∈[1,16] m, |y|≤1.15 m 的行列范围"""
    r0 = int((1.0 - GX0) / RES)
    r1 = int((16.0 - GX0) / RES)
    c0 = int((-1.15 - GY0) / RES)
    c1 = int((1.15 - GY0) / RES)
    return r0, r1, c0, c1


R0, R1, C0, C1 = corridor_mask()

tot = dict(n=0, blocked=0, shift=0, stop=0, decisive=0, gt_hit=0)
per_scene = []

for f in sorted(glob.glob('/mnt/2T_HDD/plant2_vector/vector_ds_occ/*.npz')):
    z = np.load(f, allow_mmap=True) if False else np.load(f)
    occ, gt = z['occ'], z['gt']
    ego = z['ego'] if 'ego' in z.files else None
    n = len(gt)
    st = occ[:, 2]                       # (N,64,64) 静态通道
    blocked = st[:, R0:R1, C0:C1].reshape(n, -1).any(1)
    shift = np.abs(gt[:, :, 1]).max(1) > 0.6
    if ego is not None:
        v = ego[:, 0]
    else:
        # 用 GT 位移估速度
        v = np.linalg.norm(gt[:, -1] - gt[:, 0], axis=1) / 4.0
    stop = v < 0.5
    decisive = blocked & (shift | stop)

    # GT 自身与静态占据相交（粗查：GT 点落在静态格上）
    hit = np.zeros(n, bool)
    for k in range(8):
        r = np.clip(((gt[:, k, 0] - GX0) / RES).astype(int), 0, 63)
        c = np.clip(((gt[:, k, 1] - GY0) / RES).astype(int), 0, 63)
        hit |= st[np.arange(n), r, c] > 0

    tot['n'] += n
    tot['blocked'] += int(blocked.sum())
    tot['shift'] += int(shift.sum())
    tot['stop'] += int(stop.sum())
    tot['decisive'] += int(decisive.sum())
    tot['gt_hit'] += int(hit.sum())
    per_scene.append((os.path.basename(f).replace('.npz', ''), n,
                      int(blocked.sum()), int(decisive.sum()),
                      int(shift.sum()), int(hit.sum())))

n = tot['n']
print(f'总帧数 {n}')
print(f'走廊被静态占据 (blocked): {tot["blocked"]} ({100*tot["blocked"]/n:.2f}%)')
print(f'  GT 有横移 >0.6m      : {tot["shift"]} ({100*tot["shift"]/n:.2f}%)')
print(f'  GT 停车 (v<0.5)      : {tot["stop"]} ({100*tot["stop"]/n:.2f}%)')
print(f'  **occupancy-decisive**: {tot["decisive"]} ({100*tot["decisive"]/n:.2f}%)')
print(f'GT 自身撞静态占格 (粗查): {tot["gt_hit"]} ({100*tot["gt_hit"]/n:.2f}%)')
print()
print('场景                         帧数   blocked  decisive  shift  gt_hit')
for name, nn, b, d, s, h in sorted(per_scene, key=lambda x: -x[3])[:12]:
    print(f'{name:26s} {nn:7d} {b:8d} {d:9d} {s:6d} {h:6d}')
