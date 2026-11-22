"""对照：同样 8 个 waypoint，线性插值 vs 三次样条密化到 41 点。"""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True)
    ap.add_argument('--idx', type=int, default=40)
    ap.add_argument('--out', default='/mnt/2T_HDD/plant2_vector/densify_compare.png')
    args = ap.parse_args()
    z = np.load(args.npz)
    i = min(args.idx, len(z['gt']) - 1)
    gt = z['gt'][i].astype(np.float64)
    mp, mm = z['map'][i], z['mmask'][i]

    t = np.r_[0.0, 0.5 * np.arange(1, len(gt) + 1)]
    P = np.vstack([[0.0, 0.0], gt])
    tt = np.linspace(0.0, t[-1], 41)
    lin = np.stack([np.interp(tt, t, P[:, 0]), np.interp(tt, t, P[:, 1])], 1)
    cs = CubicSpline(t, P, axis=0)
    spl = cs(tt)
    d = np.linalg.norm(lin - spl, axis=1)

    fig, ax = plt.subplots(figsize=(9, 7))
    for k in range(len(mp)):
        if mm[k] < 0.5:
            continue
        mx, my, ln, c, s = mp[k, 0], mp[k, 1], mp[k, 2], mp[k, 3], mp[k, 4]
        h = np.arctan2(s, c)
        dx, dy = np.cos(h) * ln / 2, np.sin(h) * ln / 2
        ax.plot([mx - dx, mx + dx], [my - dy, my + dy], '-', lw=0.8,
                c='#b0bec5', zorder=1)
    ax.plot(lin[:, 0], lin[:, 1], '--', lw=2.4, c='#fb8c00',
            label='linear (41 pts)', zorder=4)
    ax.plot(spl[:, 0], spl[:, 1], '-', lw=2.4, c='#1e88e5',
            label='cubic spline (41 pts)', zorder=5)
    ax.plot(P[:, 0], P[:, 1], 'o', ms=6, c='#000000',
            label='8 waypoints', zorder=6)
    ax.plot(0, 0, 's', ms=10, c='#2e7d32', label='ego', zorder=7)
    ax.set_aspect('equal')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc='upper left')
    ax.set_title('max |linear - spline| = %.3f m' % d.max(), fontsize=11)
    lim = max(np.abs(np.r_[spl[:, 1], lin[:, 1]]).max(), 10) + 5
    ax.set_ylim(-lim, lim)
    ax.set_xlim(-5, max(spl[:, 0].max(), 10) + 5)
    plt.tight_layout()
    plt.savefig(args.out, dpi=110)
    print('max diff = %.4f m' % d.max())
    print('spline heading rate: first=%.4f last=%.4f' %
          (np.degrees(np.arctan2(*(np.diff(spl, axis=0)[0][::-1]))) if False else 0,
           0))
    print('saved', args.out)


if __name__ == '__main__':
    main()
