"""对比 MLP vs Transformer 两组 600 回合训练：解析日志 + 出图 + CSV。"""
import re
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

LOGS = {'MLP (paper baseline)': '/workspace/data/logs/train_paper_mlp.log',
        'Transformer (paper recipe)': '/workspace/data/logs/train_paper_tr.log'}
OUT = '/workspace/data/compare'
os.makedirs(OUT, exist_ok=True)

pat_diag = re.compile(r'\[DIAG-ref\] median=([\d.]+) p90=([\d.]+) .*?frac<5m=(\d+)%')
pat_pdms = re.compile(r'\[PDMS\] score=([\d.]+) completion=([\d.]+)% '
                      r'collisions=(\d+) off_road=(\d+)')
pat_ep = re.compile(r'回合 (\d+)/\d+')

data = {}
for name, path in LOGS.items():
    med, p90, frac, eps, pdms, comp, coll, off = [], [], [], [], [], [], [], []
    ep = 0
    for line in open(path, encoding='utf-8', errors='ignore'):
        m = pat_ep.search(line)
        if m:
            ep = int(m.group(1))
        m = pat_diag.search(line)
        if m:
            med.append(float(m.group(1)))
            p90.append(float(m.group(2)))
            frac.append(float(m.group(3)))
            eps.append(ep)
        m = pat_pdms.search(line)
        if m:
            pdms.append(float(m.group(1)))
            comp.append(float(m.group(2)))
            coll.append(int(m.group(3)))
            off.append(int(m.group(4)))
    data[name] = dict(eps=eps, med=med, p90=p90, frac=frac,
                      pdms=pdms, comp=comp, coll=coll, off=off)
    print(name, 'epochs', len(eps), 'pdms pts', len(pdms))

# 平滑
def smooth(x, k=9):
    x = np.asarray(x, float)
    if len(x) < k:
        return x
    return np.convolve(x, np.ones(k) / k, mode='valid')

fig, axes = plt.subplots(2, 2, figsize=(11, 7), dpi=150)
for name, d in data.items():
    e = np.asarray(d['eps'], float)
    axes[0, 0].plot(e, d['med'], alpha=0.25, lw=0.6)
    axes[0, 0].plot(e[len(e) - len(smooth(d['med'])):], smooth(d['med']),
                    lw=2, label=name)
    axes[0, 1].plot(e[len(e) - len(smooth(d['frac'])):], smooth(d['frac']),
                    lw=2, label=name)
    pe = np.arange(len(d['pdms']))
    axes[1, 0].plot(pe, d['pdms'], alpha=0.25, lw=0.6)
    _s = smooth(d['pdms'], 5)
    axes[1, 0].plot(pe[:len(_s)], _s, lw=2, label=name)
    axes[1, 1].plot(pe, d['coll'], alpha=0.25, lw=0.6)
    _s2 = smooth(d['coll'], 5)
    axes[1, 1].plot(pe[:len(_s2)], _s2, lw=2, label=name)

titles = ['Tracking error (median, m)', 'frac of steps with err<5m (%)',
          'PDMS per episode', 'Collision steps per episode']
for ax, t in zip(axes.flat, titles):
    ax.set_title(t, fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
axes[0, 0].set_xlabel('epoch')
axes[0, 1].set_xlabel('epoch')
axes[1, 0].set_xlabel('evaluation index')
axes[1, 1].set_xlabel('evaluation index')
axes[0, 0].set_ylim(0, 20)
fig.suptitle('IDC controller: MLP vs Transformer (600 epochs, K=5, Q selection)')
fig.tight_layout()
fig.savefig(f'{OUT}/fig_training_curves.png')
plt.close(fig)

# 最终对比柱状图
names = list(data)
metrics = [('med', 'final median err (m)'), ('p90', 'final p90 err (m)'),
           ('pdms', 'final PDMS'), ('comp', 'completion (%)'),
           ('coll', 'collision steps'), ('off', 'off-road steps')]
fig, axes = plt.subplots(1, 6, figsize=(16, 3.2), dpi=150)
for ax, (k, title) in zip(axes, metrics):
    vals = [data[n][k][-1] for n in names]
    ax.bar(range(len(names)), vals, color=['#42a5f5', '#ef5350'])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(['MLP', 'Transf.'], fontsize=8)
    ax.set_title(title, fontsize=9)
    for i, v in enumerate(vals):
        ax.text(i, v, f'{v:.1f}', ha='center', va='bottom', fontsize=8)
    ax.grid(axis='y', alpha=0.25)
fig.suptitle('Final metrics at epoch 600 (last episode)')
fig.tight_layout()
fig.savefig(f'{OUT}/fig_final_compare.png')
plt.close(fig)

# CSV
with open(f'{OUT}/summary.csv', 'w', encoding='utf-8') as f:
    f.write('run,' + ','.join(k for k, _ in metrics) + '\n')
    for n in names:
        f.write(n + ',' + ','.join(f'{data[n][k][-1]:.3f}' for k, _ in metrics) + '\n')
print('saved', OUT)
