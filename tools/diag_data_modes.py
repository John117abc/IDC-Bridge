"""排查：塌缩是不是因为训练数据里缺少歧义场景 / 人类未来本身就没差异。

看三件事：
  1. 训练集 vs dev 的场景歧义度分布（|Δh|、路口线索、速度）是否一致
  2. "同类场景内"人类未来的分散度（横向终点位移的标准差）—— 数据本身是不是单模态
  3. k-means 锚点（20 条）本身的多样性，以及转弯场景的锚点覆盖（标签 FDE）
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, '/gpudrive/baselines/IDC-Bridge/src')
sys.path.insert(0, '/gpudrive/baselines/IDC-Bridge/tools')


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def gh(gt):
    """gt (n,8,2) → 首末段夹角（度）与终点横向位移（m，自车系 y）。"""
    d = np.diff(gt, axis=1)
    h = np.arctan2(d[..., 1], d[..., 0])
    dh = np.abs(wrap(h[:, -1] - h[:, 0])) * 180 / np.pi
    return dh, gt[:, -1, 1], gt[:, -1, 0]


def nbranch_of(z, i):
    m = int(z['n_map_r'][i])
    seg = z['map_r'][i, :m]
    lane = seg[seg[:, 4] > 0.5]
    if len(lane) == 0:
        return 0
    a, b = lane[:, :2], lane[:, 2:4]
    mid = 0.5 * (a + b)
    far = np.linalg.norm(mid, axis=1) > 15
    ang = np.abs(wrap(np.arctan2(b[:, 1] - a[:, 1], b[:, 0] - a[:, 0])))
    return int(np.sum((ang > np.pi / 6) & ~far))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/workspace/data/planner60k')
    ap.add_argument('--n-train', type=int, default=8000)
    ap.add_argument('--n-branch', type=int, default=1500)
    ap.add_argument('--anchors', default='')
    ap.add_argument('--gt-only', action='store_true',
                    help='只读 gt/ego（不解压 map_r），用于快速统计训练集构成')
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    z = np.load(f'{args.data}/train.npz', allow_pickle=True)
    N = len(z['gt'])
    rng = np.random.default_rng(0)
    sel = rng.choice(N, min(args.n_train, N), replace=False)

    gt = z['gt'][sel]
    dh, yend, xend = gh(gt)
    v0 = z['ego'][sel, 0]

    print('== 1. 训练集场景构成（%d 抽样 / 共 %d）%s'
          % (len(sel), N, args.tag))
    print('  |Δh|: p10 %.1f p50 %.1f p90 %.1f 度 | <5° %.0f%% | >20° %.0f%%'
          % (np.percentile(dh, 10), np.median(dh), np.percentile(dh, 90),
             100 * np.mean(dh < 5), 100 * np.mean(dh > 20)))
    print('  速度: p50 %.1f m/s | <1 m/s（静止/堵车）%.0f%%'
          % (np.median(v0), 100 * np.mean(v0 < 1)))
    if not args.gt_only:
        bs = rng.choice(len(sel), min(args.n_branch, len(sel)), replace=False)
        nb = np.array([nbranch_of(z, int(sel[t])) for t in bs])
        print('  路口线索（>3 条横向车道）: %.0f%%（dev 是 86%%）'
              % (100 * np.mean(nb > 3)))

    print('\n== 2. 同类场景内人类未来的分散度（数据本身是不是单模态）==')
    # 分桶：速度 × 是否转弯场景（用 GT 的 |Δh| 作为场景难度代理）
    for lo, hi, tag in ((0, 2, '静止/极慢 (<2 m/s)'), (2, 8, '中速 2-8'),
                        (8, 99, '快 (>8)')):
        m = (v0 >= lo) & (v0 < hi)
        if m.sum() < 20:
            continue
        for t2, m2 in (('直行', dh[m] < 5), ('转弯', dh[m] > 20)):
            mm = m.copy()
            mm[m] = m2
            if mm.sum() < 15:
                continue
            print('  %-16s %s n=%-4d 终点横向位移 p10/p50/p90 = %5.1f/%5.1f/%5.1f m'
                  '  std %.2f' % (tag, t2, int(mm.sum()),
                                 *np.percentile(yend[mm], [10, 50, 90]),
                                 np.std(yend[mm])))
    print('  → 若每桶内 std 只有零点几米，说明"该场景下人类也只有一种走法"，'
          '多模态无意义')

    print('\n== 3. 锚点（k-means）本身 ==')
    ap_path = args.anchors or f'{args.data}/anchors.npy'
    A = np.load(ap_path)
    K = A.shape[0]
    adh, ay, ax = gh(A.reshape(K, -1, 2))
    print('  锚点 %d 条：|Δh| %s 度' % (K, np.round(adh, 1)))
    print('  锚点终点横向范围 %.1f ~ %.1f m，纵向 %.1f ~ %.1f m'
          % (ay.min(), ay.max(), ax.min(), ax.max()))
    iu = np.triu_indices(K, 1)
    pa = np.linalg.norm(A[:, -1][iu[0]] - A[:, -1][iu[1]], axis=-1)
    print('  锚点两两终点距离: 最小 %.1f 中位 %.1f 最大 %.1f m'
          % (pa.min(), np.median(pa), pa.max()))
    print('  锚点里"转弯"（|Δh|>20°）的条数: %d / %d'
          % (int((adh > 20).sum()), K))

    # 标签残差：转弯场景是否更难被锚点覆盖
    fde = np.linalg.norm(A[None, :, -1] - gt[:, None, -1], axis=-1)  # (n,K)
    best = fde.min(1)
    print('  标签 FDE（最近的锚点）: 直行 p50 %.2f 转弯 p50 %.2f m'
          % (np.median(best[dh < 5]), np.median(best[dh > 20])))


if __name__ == '__main__':
    main()
