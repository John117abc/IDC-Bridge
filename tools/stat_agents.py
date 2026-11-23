"""统计：8 辆是否够用——溢出比例 + 被挤掉的危险车比例。

数据来源：/mnt/2T_HDD/plant2_vector/vector_ds_hist/*.npz
  agent (T,32,28) 前 8 维 = [rel_x, rel_y, cos(dh), sin(dh), v, len, wid, valid]
  已按距离升序排列（构建时 rows.sort），所以 index>=8 就是被挤掉的
"""
import glob
import sys

import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else '/mnt/2T_HDD/plant2_vector/vector_ds_hist'
files = sorted(glob.glob(D + '/*.npz'))
print('文件数', len(files))

cnt = 0
over8 = 0
over12 = 0
danger_lost = 0          # 第 9 辆及以后有“危险车”的帧
danger_lost_ttc = 0      # 其中 TTC < 2 s
n_agents = []
max_rel = []

for fp in files:
    z = np.load(fp)
    ag, am = z['agent'], z['amask']
    T = ag.shape[0]
    for t in range(0, T, 3):            # 每 3 帧取一帧，够统计
        m = am[t] > 0.5
        n = int(m.sum())
        n_agents.append(n)
        cnt += 1
        if n > 8:
            over8 += 1
        if n > 12:
            over12 += 1
        if n <= 8:
            continue
        A = ag[t][m]
        # 前 8 辆（按构建时的距离升序）
        rest = A[8:]
        rx, ry, v = rest[:, 0], rest[:, 1], rest[:, 4]
        d = np.hypot(rx, ry)
        # 危险定义：距离 < 12 m 且 速度朝向自车（近似用 |v| 与朝内分量判断）
        # 用相对位置与速度方向夹角近似：这里用简单判据——距离近 + 车速 > 2 m/s
        near = (d < 12) & (v > 2.0)
        if near.any():
            danger_lost += 1
            # TTC 粗略估计：距离 / 相对速度（用自车速度 8 m/s 近似）
            ttc = d[near] / np.maximum(v[near] + 8.0, 1.0)
            if (ttc < 2.0).any():
                danger_lost_ttc += 1

    if cnt > 40000:
        break

n_agents = np.asarray(n_agents)
print(f'\n统计帧数 = {cnt}')
print(f'周车数: 均值 {n_agents.mean():.1f}  中位 {np.median(n_agents):.0f} '
      f'p90 {np.percentile(n_agents, 90):.0f}  max {n_agents.max()}')
print(f'>8 辆的帧: {over8} ({over8/max(cnt,1)*100:.1f}%)')
print(f'>12 辆的帧: {over12} ({over12/max(cnt,1)*100:.1f}%)')
print(f'第 9 辆及以后存在“近距离有速度车”的帧: {danger_lost} '
      f'({danger_lost/max(cnt,1)*100:.1f}%)')
print(f'其中 TTC<2s 的帧: {danger_lost_ttc} ({danger_lost_ttc/max(cnt,1)*100:.1f}%)')
