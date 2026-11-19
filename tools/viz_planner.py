"""可视化：验证集上 K 条候选轨迹 vs 专家轨迹（自车系）。"""
import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, '/gpudrive/baselines/IDC-Bridge/src')
sys.path.insert(0, '/gpudrive/baselines/IDC-Bridge/tools')
from train_planner import Planner  # noqa

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COL = ['#d50000', '#2962ff', '#00c853', '#ff6d00', '#aa00ff', '#00bfa5']


def densify(a, n=40):
    t_src = np.linspace(0, 4.0, len(a))
    t_dst = np.linspace(0, 4.0, n)
    return np.stack([np.interp(t_dst, t_src, a[:, 0]),
                     np.interp(t_dst, t_src, a[:, 1])], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/workspace/data/planner5k')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', default='/workspace/data/viz_planner.png')
    ap.add_argument('--n', type=int, default=4)
    ap.add_argument('--seed', type=int, default=3)
    args = ap.parse_args()
    dev = 'cpu'
    z = np.load(f'{args.data}/val.npz', allow_pickle=True)
    ck = torch.load(args.ckpt, map_location='cpu')
    model = Planner(K=ck.get('K', 6)).to(dev).eval()
    model.load_state_dict(ck['model'])
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(z['ego']), args.n, replace=False)

    fig, axes = plt.subplots(1, args.n, figsize=(5.2 * args.n, 5.2), dpi=140)
    if args.n == 1:
        axes = [axes]
    for ax, i in zip(axes, idx):
        ego = torch.tensor(z['ego'][i:i + 1], dtype=torch.float32)
        ag = torch.tensor(z['agent'][i:i + 1], dtype=torch.float32)
        am = torch.tensor(z['amask'][i:i + 1], dtype=torch.float32)
        mp = torch.tensor(z['map'][i:i + 1], dtype=torch.float32)
        mm = torch.tensor(z['mmask'][i:i + 1], dtype=torch.float32)
        with torch.no_grad():
            pred = model(ego, ag, am, mp, mm)[0].numpy()
        gt = z['gt'][i]
        m = z['map'][i]
        mask = z['mmask'][i]
        for j in range(len(m)):
            if mask[j] <= 0:
                continue
            x, y, ln, c, s, is_lane, _ = m[j]
            th = np.arctan2(s, c)
            a = np.array([x - 0.5 * ln * c, y - 0.5 * ln * s])
            b = np.array([x + 0.5 * ln * c, y + 0.5 * ln * s])
            col = '#cfd8dc' if is_lane > 0.5 else '#ff8a65'
            ax.plot([a[0], b[0]], [a[1], b[1]], color=col, lw=1.0, zorder=1)
        g40 = densify(gt)
        ax.plot(g40[:, 0], g40[:, 1], 'k--', lw=2.2, label='expert', zorder=4)
        for k in range(pred.shape[0]):
            p40 = densify(pred[k])
            fde = np.linalg.norm(pred[k][-1] - gt[-1])
            ax.plot(p40[:, 0], p40[:, 1], color=COL[k % len(COL)],
                    lw=2.0, alpha=0.9, zorder=3,
                    label=f'mode{k} FDE={fde:.1f}')
        ax.plot(0, 0, 'k*', ms=14, zorder=5)
        ax.set_aspect('equal')
        ax.set_title(f'scene {i}', fontsize=9)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=6, loc='lower left')
        ax.set_xlabel('x (m, ego frame)')
    axes[0].set_ylabel('y (m, ego frame)')
    fig.suptitle('Planner predictions vs expert (6 modes, ego frame)')
    fig.tight_layout()
    fig.savefig(args.out)
    print('saved', args.out)


if __name__ == '__main__':
    main()
