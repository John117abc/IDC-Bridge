"""route 生成可视化（只画自车附近 120 m，过滤 (-10000,-10000) 哨兵）。"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, '/gpudrive/baselines/IDC-Bridge/tools')
from export_route import build_route, _lanes  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='/data/data_json/training')
    ap.add_argument('--names', default='')
    ap.add_argument('--out', default='/workspace/data/route_smoke.png')
    ap.add_argument('--rad', type=float, default=90.0)
    args = ap.parse_args()
    names = [x for x in args.names.split(',') if x]
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(5.6 * n, 5.6))
    if n == 1:
        axes = [axes]
    for ax, nm in zip(axes, names):
        d = json.load(open(os.path.join(args.data_dir, nm)))
        _r = build_route(d)
        route, gt = _r[0], _r[1]
        p = gt[10]
        for L in _lanes(d):
            m = np.linalg.norm(L - p[None], axis=1) < args.rad
            if m.sum() >= 2:
                ax.plot(L[:, 0], L[:, 1], '-', lw=0.7,
                        c='#78909c', zorder=1)
        for o in d['objects']:
            q = o['position'][10]
            if abs(q['x']) < 5000 and abs(q['y']) < 5000:
                if np.linalg.norm(np.array([q['x'], q['y']]) - p) < args.rad:
                    ax.plot(q['x'], q['y'], 's', ms=3.5, c='#ffb74d', zorder=2)
        ax.plot(gt[10:91, 0], gt[10:91, 1], '--', lw=1.5, c='#263238',
                label='GT future (label only)', zorder=3)
        if route is not None:
            ax.plot(route[:, 0], route[:, 1], '-', lw=3.2, c='#d50000',
                    label='route (%d pts, %.0f m)' % (len(route),
                     float(np.linalg.norm(np.diff(route, axis=0), axis=1).sum())), zorder=6)
        ax.plot(p[0], p[1], 'o', ms=9, c='#2962ff', label='ego @t=10', zorder=7)
        ax.set_xlim(p[0] - 70, p[0] + 70)
        ax.set_ylim(p[1] - 70, p[1] + 70)
        ax.set_aspect('equal')
        ax.set_title(nm.replace('tfrecord-00000-of-01000_', ''), fontsize=10)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(args.out, dpi=110)
    print('saved', args.out)


if __name__ == '__main__':
    main()
