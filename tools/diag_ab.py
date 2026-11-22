"""A/B 差异诊断：模态弧长、退化模态数、p90 minFDE 变差的来源。"""
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


def predict(ckpt, z, n, dev):
    ck = torch.load(ckpt, map_location='cpu')
    m = Planner(K=ck.get('K', 6)).to(dev).eval()
    m.load_state_dict(ck['model'])
    out = []
    with torch.no_grad():
        for b0 in range(0, n, 64):
            sl = slice(b0, min(b0 + 64, n))
            out.append(m(torch.tensor(z['ego'][sl], dtype=torch.float32, device=dev),
                         torch.tensor(z['agent'][sl], dtype=torch.float32, device=dev),
                         torch.tensor(z['amask'][sl], dtype=torch.float32, device=dev),
                         torch.tensor(z['map'][sl], dtype=torch.float32, device=dev),
                         torch.tensor(z['mmask'][sl], dtype=torch.float32, device=dev)
                         ).cpu().numpy())
    return np.concatenate(out)


def arc_len(p):
    d = np.linalg.norm(np.diff(p, axis=-2), axis=-1)
    return d.sum(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/workspace/data/planner60k')
    ap.add_argument('--a', default='/workspace/data/ab60k_A/latest.pt')
    ap.add_argument('--b', default='/workspace/data/ab60k_B/latest.pt')
    ap.add_argument('--ea', default='/workspace/data/ab60k_A/eval.npz')
    ap.add_argument('--eb', default='/workspace/data/ab60k_B/eval.npz')
    ap.add_argument('--limit', type=int, default=200)
    ap.add_argument('--out', default='/workspace/data/ab60k/diag.png')
    args = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    z = np.load(f'{args.data}/val.npz', allow_pickle=True)
    n = min(len(z['ego']), args.limit)
    gt = z['gt'][:n]
    pa = predict(args.a, z, n, dev)
    pb = predict(args.b, z, n, dev)

    la, lb = arc_len(pa), arc_len(pb)                       # (n,K)
    gt_arc = arc_len(gt)
    print('真实轨迹弧长（8 点折线）: p50 %.1f m' % np.median(gt_arc))
    print('A 模态弧长: p50 %.1f  退化(<2m) 比例 %.0f%%'
          % (np.median(la), 100 * np.mean(la < 2)))
    print('B 模态弧长: p50 %.1f  退化(<2m) 比例 %.0f%%'
          % (np.median(lb), 100 * np.mean(lb < 2)))
    dead_a = (la < 2).sum(1)
    dead_b = (lb < 2).sum(1)
    print('每场景退化模态数: A 均值 %.2f (全废 %d 场景)  B 均值 %.2f (全废 %d 场景)'
          % (dead_a.mean(), int((dead_a == la.shape[1]).sum()),
             dead_b.mean(), int((dead_b == lb.shape[1]).sum())))

    fa = np.linalg.norm(pa - gt[:, None], axis=-1)[:, :, -1].min(1)
    fb = np.linalg.norm(pb - gt[:, None], axis=-1)[:, :, -1].min(1)
    d = fb - fa
    print('minFDE: A p50 %.2f p90 %.2f   B p50 %.2f p90 %.2f'
          % (np.median(fa), np.percentile(fa, 90), np.median(fb), np.percentile(fb, 90)))
    print('B 变差 (>5 m) 场景 %d 个，其中 GT 弧长 p50 %.1f、退化模态数 B %.1f / A %.1f'
          % (int((d > 5).sum()),
             np.median(gt_arc[d > 5]) if (d > 5).any() else -1,
             dead_b[d > 5].mean() if (d > 5).any() else -1,
             dead_a[d > 5].mean() if (d > 5).any() else -1))
    print('B 变好 (<-5 m) 场景 %d 个' % int((d < -5).sum()))
    r = np.corrcoef(la.mean(1), lb.mean(1))[0, 1]
    print('逐场景平均弧长相关 A vs B: %.3f；平均弧长 A %.1f → B %.1f m'
          % (r, la.mean(), lb.mean()))

    E = np.load(args.ea)
    F = np.load(args.eb)
    ok = np.isfinite(E['pdms']).all(1) & np.isfinite(F['pdms']).all(1)
    dp = F['pdms'].max(1)[ok] - E['pdms'].max(1)[ok]
    dmin = (fb - fa)[ok]
    print('Δ(oracle PDMS) 与 Δ(minFDE) 相关: %.3f' % np.corrcoef(dp, dmin)[0, 1])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    ax.hist(la.ravel(), bins=40, alpha=.5, color='tab:blue', label=f'A p50={np.median(la):.1f}')
    ax.hist(lb.ravel(), bins=40, alpha=.5, color='tab:red', label=f'B p50={np.median(lb):.1f}')
    ax.axvline(np.median(gt_arc), color='k', ls='--', label=f'GT {np.median(gt_arc):.1f}')
    ax.set_xlabel('mode arc length (m, 4 s)'); ax.set_ylabel('#mode-scene')
    ax.set_title('mode length'); ax.legend(); ax.grid(alpha=.3)

    ax = axes[1]
    w = 0.35
    ax.bar(np.arange(7) - w / 2, np.bincount(dead_a, minlength=7) / n,
           width=w, color='tab:blue', label='A')
    ax.bar(np.arange(7) + w / 2, np.bincount(dead_b, minlength=7) / n,
           width=w, color='tab:red', label='B')
    ax.set_xlabel('#degenerate modes (<2 m) per scene'); ax.set_ylabel('fraction')
    ax.set_title('dead modes'); ax.legend(); ax.grid(alpha=.3, axis='y')

    ax = axes[2]
    ax.scatter(fa, fb, s=12, alpha=.6)
    lim = [0, max(fa.max(), fb.max()) * 1.05]
    ax.plot(lim, lim, 'k--', lw=1)
    ax.set_xlabel('A minFDE (m)'); ax.set_ylabel('B minFDE (m)')
    ax.set_title('per-scene minFDE'); ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
