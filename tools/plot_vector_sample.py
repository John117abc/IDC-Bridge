"""可视化一条向量样本（ego 系）：地图折线 + 周车框 + 自车 + GT。"""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon


def rect(x, y, h, L, W):
    c, s = np.cos(h), np.sin(h)
    pts = np.array([[-L / 2, -W / 2], [L / 2, -W / 2],
                    [L / 2, W / 2], [-L / 2, W / 2]])
    R = np.array([[c, -s], [s, c]])
    return (pts @ R.T) + np.array([x, y])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True)
    ap.add_argument('--idx', type=int, default=40)
    ap.add_argument('--out', default='/mnt/2T_HDD/plant2_vector/sample_view.png')
    args = ap.parse_args()
    z = np.load(args.npz)
    i = min(args.idx, len(z['gt']) - 1)
    mp, mm = z['map'][i], z['mmask'][i]
    ag, am = z['agent'][i], z['amask'][i]
    gt = z['gt'][i]

    fig, ax = plt.subplots(figsize=(8, 8))
    for k in range(len(mp)):
        if mm[k] < 0.5:
            continue
        mx, my, ln, c, s = mp[k, 0], mp[k, 1], mp[k, 2], mp[k, 3], mp[k, 4]
        h = np.arctan2(s, c)
        dx, dy = np.cos(h) * ln / 2, np.sin(h) * ln / 2
        ax.plot([mx - dx, mx + dx], [my - dy, my + dy], '-', lw=0.8,
                c='#78909c', zorder=1)
    for k in range(len(ag)):
        if am[k] < 0.5:
            continue
        x, y, dh, ln, wd = ag[k, 0], ag[k, 1], np.arctan2(ag[k, 3], ag[k, 2]), ag[k, 5], ag[k, 6]
        ax.add_patch(Polygon(rect(x, y, dh, max(ln, 1.0), max(wd, 1.0)),
                             closed=True, fc='#ffb74d', ec='#e65100', lw=0.8,
                             alpha=0.9, zorder=3))
        vx, vy = np.cos(dh) * ag[k, 4] * 0.5, np.sin(dh) * ag[k, 4] * 0.5
        ax.plot([x, x + vx], [y, y + vy], '-', c='#bf360c', lw=1.2, zorder=4)
    ax.add_patch(Polygon(rect(0, 0, 0, 4.8, 1.9), closed=True,
                         fc='#2962ff', ec='#0d47a1', lw=1.0, zorder=6))
    _g = np.vstack([[0.0, 0.0], gt])   # 从自车原点画起（gt 是 0.5..4.0 s，不含 0 s）
    ax.plot(_g[:, 0], _g[:, 1], '--o', ms=3, lw=1.8, c='#000000',
            label='GT 0~4 s', zorder=7)
    ax.plot(_g[1:-1, 0], _g[1:-1, 1], 'o', ms=4, c='#000000', zorder=7)
    ax.plot(0, 0, 'o', ms=6, c='#2962ff', zorder=8)
    ax.set_xlim(-25, 45)
    ax.set_ylim(-30, 30)
    ax.set_aspect('equal')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc='upper right')
    ax.set_title('frame %d  agents=%d  map_segs=%d' %
                 (i, int(am.sum()), int(mm.sum())), fontsize=10)
    plt.tight_layout()
    plt.savefig(args.out, dpi=110)
    print('saved', args.out)


if __name__ == '__main__':
    main()
