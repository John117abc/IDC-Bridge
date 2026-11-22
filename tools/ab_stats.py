"""A/B 配对统计：oracle PDMS、逐模态提升、符号检验、bootstrap CI。"""
import argparse

import numpy as np


def boot_ci(x, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    m = np.array([np.mean(rng.choice(x, len(x), replace=True)) for _ in range(n)])
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', default='/workspace/data/ab60k_A/eval.npz')
    ap.add_argument('--b', default='/workspace/data/ab60k_B/eval.npz')
    args = ap.parse_args()
    A = np.load(args.a)
    B = np.load(args.b)
    pa, pb = A['pdms'], B['pdms']
    ok = np.isfinite(pa).all(1) & np.isfinite(pb).all(1)
    pa, pb = pa[ok], pb[ok]
    n, K = pa.shape
    print(f'配对场景 {n}  模态 {K}')

    oa, ob = pa.max(1), pb.max(1)
    d = ob - oa
    lo, hi = boot_ci(d)
    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    print(f'oracle PDMS  A {oa.mean():.1f}  B {ob.mean():.1f}  '
          f'Δ {d.mean():+.1f} CI[{lo:+.1f},{hi:+.1f}]')
    print(f'  中位 A {np.median(oa):.1f}  B {np.median(ob):.1f}')
    print(f'  符号检验: B 更好 {pos} / 更差 {neg} / 持平 {n-pos-neg} '
          f'(p≈{min(1.0, 2*min(pos,neg)/max(pos+neg,1)):.3f} 二项近似)')

    print('逐模态 PDMS 均值 / 中位 / >0 比例:')
    for k in range(K):
        a, b = pa[:, k], pb[:, k]
        print(f'  mode {k}: A {a.mean():5.1f}/{np.median(a):5.1f}/'
              f'{100*np.mean(a > 0):3.0f}%   '
              f'B {b.mean():5.1f}/{np.median(b):5.1f}/{100*np.mean(b > 0):3.0f}%'
              f'   Δ {b.mean()-a.mean():+5.1f}')
    allp = slice(None)
    print(f'全模态 PDMS 均值: A {pa[allp].mean():.1f}  B {pb[allp].mean():.1f}'
          f'  Δ {pb.mean()-pa.mean():+.1f}')
    # 逐场景"全模态均值"的配对差（每个场景等权）
    ma, mb = pa.mean(1), pb.mean(1)
    dm = mb - ma
    lo2, hi2 = boot_ci(dm)
    print(f'逐场景全模态均值: A {ma.mean():.1f}  B {mb.mean():.1f}  '
          f'Δ {dm.mean():+.1f} CI[{lo2:+.1f},{hi2:+.1f}]  '
          f'B 更好 {int((dm>0).sum())} / 更差 {int((dm<0).sum())}')
    print(f'PDMS>0 占比: A {100*np.mean(pa>0):.0f}%  B {100*np.mean(pb>0):.0f}%')
    print(f'PDMS>60 占比: A {100*np.mean(pa>60):.0f}%  B {100*np.mean(pb>60):.0f}%')
    print(f'PDMS=0 占比: A {100*np.mean(pa<=0):.0f}%  B {100*np.mean(pb<=0):.0f}%')

    fa, fb = A['fde'].min(1)[ok], B['fde'].min(1)[ok]
    print(f'minFDE p50: A {np.median(fa):.2f}  B {np.median(fb):.2f}')
    print(f'minFDE p90: A {np.percentile(fa,90):.2f}  B {np.percentile(fb,90):.2f}')
    print(f'minFDE<2m : A {100*np.mean(fa<2):.0f}%  B {100*np.mean(fb<2):.0f}%')

    # 每个模态"排名第一"的场景数（哪条模态最常赢）
    print('oracle 选中的模态分布:')
    print('  A:', np.bincount(pa.argmax(1), minlength=K).tolist())
    print('  B:', np.bincount(pb.argmax(1), minlength=K).tolist())


if __name__ == '__main__':
    main()
