"""统计"真正存在多模态"的场景子集有多大（训练集 & dev），用于限定多样性评估的桶。"""
import argparse
import sys

import numpy as np

sys.path.insert(0, '/gpudrive/baselines/IDC-Bridge/tools')
from diag_data_modes import gh, nbranch_of  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/workspace/data/planner60k')
    ap.add_argument('--split', default='val')
    ap.add_argument('--n', type=int, default=6000)
    ap.add_argument('--nbranch', type=int, default=0, help='>0 时额外统计路口线索')
    args = ap.parse_args()
    z = np.load(f'{args.data}/{args.split}.npz', allow_pickle=True)
    n = min(len(z['gt']), args.n)
    idx = np.arange(n)
    gt = z['gt'][idx]
    dh, _, _ = gh(gt)
    v0 = z['ego'][idx, 0]
    moving = v0 > 2.0
    print('== %s 前 %d 条 ==' % (args.split, n))
    print('  速度>2 m/s: %.0f%%' % (100 * np.mean(moving)))
    print('  GT |Δh|>20° 且 速度>2（真·多模态候选）: %.0f%% (%d 条)'
          % (100 * np.mean(moving & (dh > 20)), int(np.sum(moving & (dh > 20)))))
    print('  GT |Δh|>10° 且 速度>2: %.0f%% (%d 条)'
          % (100 * np.mean(moving & (dh > 10)), int(np.sum(moving & (dh > 10)))))
    if args.nbranch > 0:
        sub = np.random.default_rng(0).choice(n, min(args.nbranch, n), replace=False)
        nb = np.array([nbranch_of(z, int(i)) for i in sub])
        mm = moving[sub] & (nb > 3)
        print('  速度>2 且 路口线索(nbranch>3): %d/%d 抽样里 %d 条'
              % (len(sub), n, int(mm.sum())))


if __name__ == '__main__':
    main()
