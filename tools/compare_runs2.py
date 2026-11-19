"""对比 G1(MLP) vs G2(Transformer)：75 worlds, 500 epochs, 三段课程 + 每步Q选路。"""
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

LOGS = [('MLP (paper baseline)', '/workspace/data/logs/g1_mlp.log', '#1e88e5'),
        ('Transformer (ours)', '/workspace/data/logs/g2_tr.log', '#e53935')]
OUT = '/workspace/data/compare2'
os.makedirs(OUT, exist_ok=True)

pat_diag = re.compile(r'\[DIAG-ref\] median=([\d.]+) p90=([\d.]+) .*?frac<5m=(\d+)%')
pat_pdms = re.compile(r'\[PDMS\] score=([\d.]+) completion=([\d.]+)% '
                      r'collisions=(\d+) off_road=(\d+)')
pat_cl = re.compile(r'Critic Loss:\s*([\d.]+), Actor Loss:\s*(-|[\d.]+)')
pat_ep = re.compile(r'回合 (\d+)/')

data = {}
for name, path, color in LOGS:
    d = dict(ep=[], med=[], p90=[], frac=[], pdms=[], comp=[], coll=[],
             off=[], cl=[], al=[], color=color)
    ep = 0
    for line in open(path, encoding='utf-8', errors='ignore'):
        m = pat_ep.search(line)
        if m:
            ep = int(m.group(1))
        m = pat_diag.search(line)
        if m:
            d['ep'].append(ep)
            d['med'].append(float(m.group(1)))
            d['p90'].append(float(m.group(2)))
            d['frac'].append(float(m.group(3)))
        m = pat_pdms.search(line)
        if m:
            d['pdms'].append(float(m.group(1)))
            d['comp'].append(float(m.group(2)))
            d['coll'].append(int(m.group(3)))
            d['off'].append(int(m.group(4)))
        m = pat_cl.search(line)
        if m:
            d['cl'].append(float(m.group(1)))
            if m.group(2) != '-':
                d['al'].append(float(m.group(2)))
    data[name] = d
    print(name, 'diag', len(d['ep']), 'pdms', len(d['pdms']), 'loss', len(d['cl']))


def smooth(x, k=25):
    x = np.asarray(x, float)
    if len(x) < k:
        return x
    return np.convolve(x, np.ones(k) / k, mode='valid')


fig, axes = plt.subplots(2, 2, figsize=(12, 7), dpi=150)
for name, d in data.items():
    e = np.asarray(d['ep'], float)
    axes[0, 0].plot(e, d['med'], alpha=0.2, lw=0.6, color=d['color'])
    s = smooth(d['med'])
    axes[0, 0].plot(e[len(e) - len(s):], s, lw=2, color=d['color'], label=name)
    s2 = smooth(d['frac'])
    axes[0, 1].plot(e[len(e) - len(s2):], s2, lw=2, color=d['color'], label=name)

    pe = np.arange(len(d['pdms']))
    axes[1, 0].plot(pe, d['pdms'], alpha=0.2, lw=0.6, color=d['color'])
    s3 = smooth(d['pdms'], 10)
    axes[1, 0].plot(pe[:len(s3)], s3, lw=2, color=d['color'], label=name)
    axes[1, 0].plot(pe, d['comp'], '--', alpha=0.35, lw=1, color=d['color'])

    s4 = smooth(d['coll'], 10)
    axes[1, 1].plot(pe[:len(s4)], s4, lw=2, color=d['color'], label=name + ' coll')
    s5 = smooth(d['off'], 10)
    axes[1, 1].plot(pe[:len(s5)], s5, '--', lw=1.4, color=d['color'],
                    label=name + ' off-road')

for ax, t in zip(axes.flat, ['Tracking error median (m)',
                             'Fraction of worlds with err<5m (%)',
                             'PDMS (solid) & completion % (dashed)',
                             'Collisions (solid) / off-road (dashed) steps per episode']):
    ax.set_title(t, fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    ax.set_xlabel('epoch')
axes[1, 0].set_ylim(0, 100)
axes[1, 1].set_yscale('log')
fig.suptitle('IDC controller: MLP vs Transformer (75 worlds, 500 epochs, K=5 + Q selection)')
fig.tight_layout()
fig.savefig(f'{OUT}/fig_compare_curves.png')
plt.close(fig)

# 最终指标柱状
names = list(data)
keys = [('pdms', 'PDMS'), ('comp', 'completion %'), ('coll', 'collisions'),
        ('off', 'off-road'), ('med', 'median err (m)'), ('frac', 'frac<5m %')]
fig, axes = plt.subplots(1, 6, figsize=(16, 3.2), dpi=150)
for ax, (k, title) in zip(axes, keys):
    vals = [data[n][k][-1] for n in names]
    ax.bar([0, 1], vals, color=[data[n]['color'] for n in names])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['MLP', 'Transf.'], fontsize=8)
    ax.set_title(title, fontsize=9)
    for i, v in enumerate(vals):
        ax.text(i, v, f'{v:.1f}', ha='center', va='bottom', fontsize=8)
    ax.grid(axis='y', alpha=0.25)
fig.suptitle('Final metrics at epoch 500 (75 worlds)')
fig.tight_layout()
fig.savefig(f'{OUT}/fig_compare_final.png')
plt.close(fig)

with open(f'{OUT}/summary.csv', 'w', encoding='utf-8') as f:
    f.write('run,' + ','.join(k for k, _ in keys) + '\n')
    for n in names:
        f.write(n + ',' + ','.join(f'{data[n][k][-1]:.3f}' for k, _ in keys) + '\n')
print('saved', OUT)
