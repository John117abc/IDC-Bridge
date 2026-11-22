"""查 B 组"全模态退化（<2m）"的场景：是合理停车还是塌缩。"""
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
    ap.add_argument('--limit', type=int, default=200)
    args = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    z = np.load(f'{args.data}/val.npz', allow_pickle=True)
    n = min(len(z['ego']), args.limit)
    gt = z['gt'][:n]
    pa, pb = predict(args.a, z, n, dev), predict(args.b, z, n, dev)
    la, lb = arc_len(pa), arc_len(pb)
    ga = arc_len(gt)
    v0 = z['ego'][:n, 0]
    nag = z['amask'][:n].sum(1)

    all_dead = (lb < 2).all(1)
    print('B 全模态退化场景 %d / %d' % (int(all_dead.sum()), n))
    print('  这些场景: GT 弧长 p50 %.1f (全体 %.1f)  自车速度 p50 %.1f (全体 %.1f)'
          % (np.median(ga[all_dead]), np.median(ga), np.median(v0[all_dead]),
             np.median(v0)))
    print('  A 在这些场景的平均弧长 %.1f，A 也有退化模态的比例 %.0f%%'
          % (la[all_dead].mean(), 100 * np.mean(la[all_dead] < 2)))
    print('  GT 弧长 < 3 m 的场景数: %d（即合理停车）' % int((ga[all_dead] < 3).sum()))
    print('\n场景 | GT弧长 | v0 | 周车 | A弧长均值 | B弧长均值 | A minFDE | B minFDE')
    fa = np.linalg.norm(pa - gt[:, None], axis=-1)[:, :, -1].min(1)
    fb = np.linalg.norm(pb - gt[:, None], axis=-1)[:, :, -1].min(1)
    idx = np.where(all_dead)[0]
    for i in idx[:20]:
        print('%4d | %6.1f | %4.1f | %3d | %8.1f | %8.1f | %7.2f | %7.2f'
              % (i, ga[i], v0[i], nag[i], la[i].mean(), lb[i].mean(), fa[i], fb[i]))


if __name__ == '__main__':
    main()
