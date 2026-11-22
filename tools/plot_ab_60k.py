"""画 60k A/B 对比图：训练曲线 + PDMS/minFDE 分布。"""
import argparse
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

LOG = re.compile(r'ep\s+(\d+)\s+wta=([\d.]+)')
IA = re.compile(r'iawr=([\d.]+)\s+upd=(\d+).*?R=([\d.]+)\[([-\d.]+),([-\d.]+)\]'
                r'\s+Rstd=([\d.]+)\s+std0=(\d+)%\s+pos=(\d+)%')


def parse(path):
    wta, ia = [], []
    for ln in open(path, errors='replace'):
        m = LOG.search(ln)
        if m:
            wta.append((int(m.group(1)), float(m.group(2))))
        m = IA.search(ln)
        if m:
            ia.append((int(re.search(r'ep\s+(\d+)', ln).group(1)),) +
                      tuple(float(x) for x in m.groups()))
    return np.array(wta), np.array(ia)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a-log', default='/workspace/data/logs/ab60k_A.log')
    ap.add_argument('--b-log', default='/workspace/data/logs/ab60k_B.log')
    ap.add_argument('--a-eval', default='/workspace/data/ab60k_A/eval.npz')
    ap.add_argument('--b-eval', default='/workspace/data/ab60k_B/eval.npz')
    ap.add_argument('--out', default='/workspace/data/ab60k/ab_curves.png')
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    for p, lab, c in ((args.a_log, 'A: WTA + div', 'tab:blue'),
                      (args.b_log, 'B: + IAWR(PDMS)', 'tab:red')):
        try:
            wta, ia = parse(p)
            ax.plot(wta[:, 0], wta[:, 1], 'o-', color=c, label=lab)
        except Exception as e:
            print(f'[skip] {p}: {e}')
    ax.set_xlabel('epoch'); ax.set_ylabel('WTA loss (L1 to GT)')
    ax.set_title('train'); ax.legend(); ax.grid(alpha=.3)

    ax = axes[1]
    a = np.load(args.a_eval)['pdms'] if args.a_eval else None
    b = np.load(args.b_eval)['pdms'] if args.b_eval else None
    data, labs, cols = [], [], []
    for arr, lab, c in ((a, 'A', 'tab:blue'), (b, 'B', 'tab:red')):
        if arr is None:
            continue
        v = arr[np.isfinite(arr)]
        data.append(v); labs.append(f'{lab}\n(n={v.size})'); cols.append(c)
    if data:
        bp = ax.boxplot(data, labels=labs, patch_artist=True, showfliers=False)
        for patch, c in zip(bp['boxes'], cols):
            patch.set_facecolor(c); patch.set_alpha(.5)
        ax.set_ylabel('PDMS per mode'); ax.set_title('executed PDMS (all modes)')
        ax.grid(alpha=.3, axis='y')

    ax = axes[2]
    for p, lab, c in ((args.a_eval, 'A', 'tab:blue'), (args.b_eval, 'B', 'tab:red')):
        try:
            z = np.load(p)
            fde = z['fde'].min(1)
            ax.hist(fde, bins=30, alpha=.5, color=c, label=f'{lab} minFDE p50={np.median(fde):.2f}')
        except Exception as e:
            print(f'[skip] {p}: {e}')
    ax.set_xlabel('minFDE (m)'); ax.set_ylabel('#scenes'); ax.set_title('minFDE')
    ax.legend(); ax.grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
