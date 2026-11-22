"""A/B 定性对比：指定场景的 6 条模态（上排 A，下排 B）。"""
import argparse
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, '/gpudrive/baselines/IDC-Bridge/src')
sys.path.insert(0, '/gpudrive/baselines/IDC-Bridge/tools')
from train_planner import Planner  # noqa: E402
from viz_planner import COL, densify  # noqa: E402


def predict(ckpt, z, idx, dev):
    ck = torch.load(ckpt, map_location='cpu')
    m = Planner(K=ck.get('K', 6)).to(dev).eval()
    m.load_state_dict(ck['model'])
    out = {}
    with torch.no_grad():
        for i in idx:
            sl = slice(i, i + 1)
            out[i] = m(torch.tensor(z['ego'][sl], dtype=torch.float32),
                       torch.tensor(z['agent'][sl], dtype=torch.float32),
                       torch.tensor(z['amask'][sl], dtype=torch.float32),
                       torch.tensor(z['map'][sl], dtype=torch.float32),
                       torch.tensor(z['mmask'][sl], dtype=torch.float32)
                       )[0].numpy()
    return out


def arc(p):
    return float(np.linalg.norm(np.diff(p, axis=-2), axis=-1).sum(-1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/workspace/data/planner60k')
    ap.add_argument('--a', default='/workspace/data/ab60k_A/latest.pt')
    ap.add_argument('--b', default='/workspace/data/ab60k_B/latest.pt')
    ap.add_argument('--scenes', default='16,30,91')
    ap.add_argument('--out', default='/workspace/data/ab60k/viz_stopped.png')
    args = ap.parse_args()
    idx = [int(x) for x in args.scenes.split(',')]
    z = np.load(f'{args.data}/val.npz', allow_pickle=True)
    dev = 'cpu'
    pa = predict(args.a, z, idx, dev)
    pb = predict(args.b, z, idx, dev)

    fig, axes = plt.subplots(2, len(idx), figsize=(5.0 * len(idx), 10), dpi=130)
    axes = np.atleast_2d(axes)
    for c, i in enumerate(idx):
        gt = z['gt'][i]
        m, mask = z['map'][i], z['mmask'][i]
        for r, (pred, tag) in enumerate(((pa[i], 'A: WTA + div'),
                                        (pb[i], 'B: + IAWR (PDMS)'))):
            ax = axes[r, c]
            for j in range(len(m)):
                if mask[j] <= 0:
                    continue
                x, y, ln, cs, sn, is_lane, _ = m[j]
                a = np.array([x - 0.5 * ln * cs, y - 0.5 * ln * sn])
                b = np.array([x + 0.5 * ln * cs, y + 0.5 * ln * sn])
                ax.plot([a[0], b[0]], [a[1], b[1]],
                        color='#cfd8dc' if is_lane > 0.5 else '#ff8a65',
                        lw=1.0, zorder=1)
            g40 = densify(gt)
            ax.plot(g40[:, 0], g40[:, 1], 'k--', lw=2.4, label='expert', zorder=4)
            for k in range(pred.shape[0]):
                p40 = densify(pred[k])
                ax.plot(p40[:, 0], p40[:, 1], color=COL[k % len(COL)],
                        lw=2.0, alpha=0.9, zorder=3, label=f'mode{k}')
            ax.plot(0, 0, 'k*', ms=14, zorder=5)
            ax.set_aspect('equal')
            ax.grid(alpha=0.2)
            ax.set_title(f'{tag}  |  scene {i}\nmean arc {arc(pred):.1f} m, '
                         f'expert {arc(gt):.1f} m', fontsize=9)
            ax.legend(fontsize=6, loc='lower left')
            ax.set_xlim(-12, 32)
            ax.set_ylim(-14, 14)
            if c == 0:
                ax.set_ylabel('y (m, ego frame)')
            if r == 1:
                ax.set_xlabel('x (m, ego frame)')
    fig.suptitle('Stopped scenes: A drives off in all 6 modes, B stops (expert moves 0-13 m)',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out)
    print('saved', args.out)


if __name__ == '__main__':
    main()

