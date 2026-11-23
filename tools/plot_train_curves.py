"""解析训练日志，画损失/指标曲线（支持多模型对比）。

用法:
  python plot_train_curves.py --logs A=/path/train_idcsel.log B=/path/train_conf.log \
      --outdir /mnt/2T_HDD/plant2_vector/curves
"""
import argparse
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PAT = re.compile(r'^ep\s*(\d+)/\d+\s+wta\s+([\d.]+)(?:\s+ce\s+([\d.]+))?\s*\|'
                 r'\s*val minADE\s+([\d.]+)\s+minFDE\s+([\d.]+)'
                 r'(?:\s+minPairDiv\s+([\d.]+))?')
PAT2 = re.compile(r'^ep\s*(\d+)/\d+\s+wta\s+([\d.]+)\s+ce\s+([\d.]+)\s*\|'
                  r'\s*val minADE\s+([\d.]+)\s+minFDE\s+([\d.]+)(?:\s+minPairDiv\s+([\d.]+))?')


def parse(fp):
    if not os.path.exists(fp):
        return None
    out = []
    for ln in open(fp, errors='ignore'):
        m = PAT.match(ln.strip()) or PAT2.match(ln.strip())
        if not m:
            continue
        g = m.groups()
        out.append(dict(ep=int(g[0]), wta=float(g[1]),
                        ce=float(g[2]) if g[2] else None,
                        ade=float(g[3]), fde=float(g[4]),
                        div=float(g[5]) if len(g) > 5 and g[5] else None))
    return out or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logs', required=True)      # name=path,name=path
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    data = {}
    for spec in args.logs.split(','):
        if '=' not in spec:
            continue
        name, fp = spec.split('=', 1)
        d = parse(fp)
        if d:
            data[name] = d
            print(f'{name}: {len(d)} epochs (last ep {d[-1]["ep"]}, '
                  f'FDE {d[-1]["fde"]:.3f})')
    if not data:
        print('没有解析到任何曲线')
        return

    cols = {'A_idcsel': '#1e88e5', 'B_conf': '#d50000'}
    for k, name in enumerate(data):
        cols.setdefault(name, ['#1e88e5', '#d50000', '#2e7d32', '#6a1b9a'][k % 4])

    def fig_one(key, ylabel, title, fname, ylog=False):
        plt.figure(figsize=(6.4, 4.2))
        for name, d in data.items():
            xs = [r['ep'] for r in d]
            ys = [r[key] for r in d]
            pairs = [(x, y) for x, y in zip(xs, ys) if y is not None]
            if not pairs:
                continue
            xs, ys = zip(*pairs)
            plt.plot(xs, ys, '-', lw=2, c=cols[name], label=name)
        plt.xlabel('epoch'); plt.ylabel(ylabel); plt.title(title)
        plt.grid(alpha=0.3); plt.legend()
        if ylog:
            plt.yscale('log')
        plt.tight_layout()
        fp = os.path.join(args.outdir, fname)
        plt.savefig(fp, dpi=120); plt.close()
        print('saved', fp)

    fig_one('wta', 'train WTA loss', 'Training loss (WTA)', 'loss_wta.png')
    fig_one('ce', 'train CE loss', 'Confidence-head CE loss', 'loss_ce.png')
    fig_one('fde', 'val minFDE (m)', 'Validation minFDE', 'val_minfde.png')
    fig_one('ade', 'val minADE (m)', 'Validation minADE', 'val_minade.png')
    fig_one('div', 'min pair distance (m)', 'Mode diversity (min pair)', 'val_div.png')

    # 合并面板
    keys = [('wta', 'train WTA'), ('fde', 'val minFDE (m)'),
            ('ade', 'val minADE (m)'), ('div', 'min pair (m)')]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, (key, ylab) in zip(axes.ravel(), keys):
        for name, d in data.items():
            xs = [r['ep'] for r in d]
            ys = [r[key] for r in d]
            pairs = [(x, y) for x, y in zip(xs, ys) if y is not None]
            if pairs:
                ax.plot(*zip(*pairs), '-', lw=2, c=cols[name], label=name)
        ax.set_xlabel('epoch'); ax.set_ylabel(ylab)
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle('PlanT2 vector planner — loss & metrics')
    plt.tight_layout()
    fp = os.path.join(args.outdir, 'summary.png')
    plt.savefig(fp, dpi=120); plt.close()
    print('saved', fp)


if __name__ == '__main__':
    main()
