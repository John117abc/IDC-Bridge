"""按"GT 会动的场景"过滤后的 A/B 对比（避免把合理停车算成退化）。"""
import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, '/gpudrive/baselines/IDC-Bridge/src')
sys.path.insert(0, '/gpudrive/baselines/IDC-Bridge/tools')
from diag_ab import arc_len, predict  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/workspace/data/planner60k')
    ap.add_argument('--a', default='/workspace/data/ab60k_A/latest.pt')
    ap.add_argument('--b', default='/workspace/data/ab60k_B/latest.pt')
    ap.add_argument('--ea', default='/workspace/data/ab60k_A/eval.npz')
    ap.add_argument('--eb', default='/workspace/data/ab60k_B/eval.npz')
    ap.add_argument('--limit', type=int, default=200)
    ap.add_argument('--gt-min', type=float, default=5.0)
    args = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    z = np.load(f'{args.data}/val.npz', allow_pickle=True)
    n = min(len(z['ego']), args.limit)
    gt = z['gt'][:n]
    pa, pb = predict(args.a, z, n, dev), predict(args.b, z, n, dev)
    la, lb = arc_len(pa), arc_len(pb)
    ga = arc_len(gt)
    moving = ga > args.gt_min
    E, F = np.load(args.ea), np.load(args.eb)

    print('场景总数 %d：会动 (GT>%.0fm) %d，几乎停车 %d'
          % (n, args.gt_min, int(moving.sum()), int((~moving).sum())))
    for tag, m in (('会动', moving), ('停车', ~moving)):
        print('--- %s 场景 (%d) ---' % (tag, int(m.sum())))
        print('  平均弧长: A %.1f  B %.1f  GT %.1f'
              % (la[m].mean(), lb[m].mean(), ga[m].mean()))
        print('  退化模态(<2m)比例: A %.0f%%  B %.0f%%'
              % (100 * np.mean(la[m] < 2), 100 * np.mean(lb[m] < 2)))
        fa = np.linalg.norm(pa[m] - gt[m][:, None], axis=-1)[:, :, -1].min(1)
        fb = np.linalg.norm(pb[m] - gt[m][:, None], axis=-1)[:, :, -1].min(1)
        print('  minFDE p50: A %.2f  B %.2f   p90: A %.2f  B %.2f'
              % (np.median(fa), np.median(fb), np.percentile(fa, 90),
                 np.percentile(fb, 90)))
        okp = np.isfinite(E['pdms'][m]).all(1) & np.isfinite(F['pdms'][m]).all(1)
        pm_a, pm_b = E['pdms'][m][okp], F['pdms'][m][okp]
        print('  全模态 PDMS: A %.1f  B %.1f  (Δ %+.1f)'
              % (pm_a.mean(), pm_b.mean(), pm_b.mean() - pm_a.mean()))
        print('  oracle PDMS: A %.1f  B %.1f' % (pm_a.max(1).mean(), pm_b.max(1).mean()))


if __name__ == '__main__':
    main()
