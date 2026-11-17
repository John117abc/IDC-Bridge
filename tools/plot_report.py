"""生成报告用图：PDMS 对比 / K 扫描 / 轨迹可视化 / 训练曲线。"""
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "tools"))

OUT = '/workspace/data/report'
os.makedirs(OUT, exist_ok=True)

# ---------- 数据（来自各次评测日志） ----------
METHODS = ['expert', 'oracle', 'head', 'const-vel']
MEAN = [100.0, 88.9, 78.9, 75.8]
P10 = [100.0, 71.4, 47.2, 0.0]
P50 = [100.0, 94.1, 89.7, 90.7]
P90 = [100.0, 98.6, 96.5, 98.9]
COLORS = ['#4caf50', '#2196f3', '#e53935', '#9e9e9e']


def fig1():
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=160)
    x = np.arange(len(METHODS))
    bars = ax.bar(x, MEAN, color=COLORS, width=0.6, alpha=0.9,
                  edgecolor='black', linewidth=0.6)
    for i in range(len(METHODS)):
        ax.vlines(x[i], P10[i], P90[i], color='black', lw=1.2, alpha=0.6)
        ax.plot(x[i], P10[i], 'v', color='black', ms=5)
        ax.plot(x[i], P90[i], '^', color='black', ms=5)
        ax.plot(x[i], P50[i], 'o', color='white', mec='black', ms=6)
        ax.text(x[i], MEAN[i] + 2, f'{MEAN[i]:.1f}', ha='center',
                fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(METHODS, fontsize=10)
    ax.set_ylabel('Open-loop PDMS', fontsize=11)
    ax.set_ylim(0, 112)
    ax.axhline(75.8, color='gray', ls=':', lw=1)
    ax.set_title('Planning head vs baselines (150 scenes, 4 s, K=20 medoid anchors)',
                 fontsize=12)
    ax.grid(axis='y', alpha=0.25)
    ax.text(0.99, 0.03, 'bar=mean  o=p50  v/^=p10/p90',
            transform=ax.transAxes, ha='right', fontsize=8, color='#555')
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig1_pdms.png')
    plt.close(fig)


def fig2():
    K = [20, 32, 64]
    head = [78.9, 68.4, 67.1]
    ora = [88.9, 89.0, 90.0]
    cov = [1.23, 1.06, 0.74]
    acc = [66.2, 57.3, 45.1]
    p10 = [47.2, 0.0, 0.0]
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=160)
    ax.plot(K, ora, 'o-', color='#2196f3', label='oracle (anchor upper bound)', lw=2)
    ax.plot(K, head, 's-', color='#e53935', label='head', lw=2)
    ax.plot(K, p10, '^--', color='#ff9800', label='head p10 (tail)', lw=1.6)
    for i, k in enumerate(K):
        ax.annotate(f'{head[i]:.1f}', (k, head[i]), textcoords='offset points',
                    xytext=(0, 8), ha='center', color='#e53935', fontsize=9)
        ax.annotate(f'{ora[i]:.1f}', (k, ora[i]), textcoords='offset points',
                    xytext=(0, -14), ha='center', color='#2196f3', fontsize=9)
    ax.set_xlabel('number of anchors K', fontsize=11)
    ax.set_ylabel('PDMS', fontsize=11)
    ax.set_xticks(K)
    ax.set_ylim(35, 100)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc='lower left')
    ax2 = ax.twinx()
    ax2.plot(K, cov, 'd:', color='#607d8b', lw=1.4, label='coverage FDE p50 (m)')
    ax2.plot(K, acc, 'x:', color='#8bc34a', lw=1.4, label='anchor accuracy (%)')
    ax2.set_ylabel('coverage (m) / accuracy (%)', fontsize=10)
    ax2.set_ylim(0, 70)
    ax2.legend(fontsize=8, loc='upper right')
    ax.set_title('Anchor-count trade-off: better coverage, worse selection', fontsize=12)
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig2_k_sweep.png')
    plt.close(fig)


def fig3(n_scenes=6):
    import torch
    from train_head import Head
    from eval_head_pdms import densify, ego_to_world
    from metrics.openloop_pdms import Scene, score_trajectory
    from build_head_dataset import build_one

    D = '/workspace/data/head_dd/medoid'
    anchors_np = np.load(os.path.join(D, 'anchors.npy'))
    K, T = anchors_np.shape[0], anchors_np.shape[1]
    ck = torch.load(os.path.join(D, 'head_best.pt'), map_location='cpu')
    model = Head(ck.get('K', K), ck.get('T', T))
    model.load_state_dict(ck['model'])
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(dev).eval()
    anchors = torch.tensor(anchors_np, dtype=torch.float32, device=dev)

    z = np.load(os.path.join(D, 'train.npz'))
    ego_m, ego_s = z['ego'].mean(0), z['ego'].std(0) + 1e-3
    am = z['amask'].astype(bool)
    ag_m, ag_s = z['agent'][am].mean(0), z['agent'][am].std(0) + 1e-3
    mm = z['mmask'].astype(bool)
    mp_m, mp_s = z['map'][mm].mean(0), z['map'][mm].std(0) + 1e-3

    files = sorted(glob.glob('/data/data_json/training/tfrecord-*.json'))
    rng = np.random.default_rng(1)
    files = [files[i] for i in rng.choice(len(files), 40, replace=False)]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=150)
    shown = 0
    for fp in files:
        if shown >= n_scenes:
            break
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        sc = Scene(d, horizon=40)
        s = build_one(d, anchors_np)
        if s is None or sc.n_future < 20:
            continue
        p0, h0 = sc.expert_pos[sc.t_now], float(sc.expert_heading[sc.t_now])
        H = sc.n_future
        with torch.no_grad():
            ego = torch.tensor((s['ego'] - ego_m) / ego_s, device=dev)[None]
            ag = torch.tensor((s['agent'] - ag_m) / ag_s, device=dev)[None]
            am_ = torch.tensor(s['amask'], device=dev)[None]
            mp_ = torch.tensor((s['map'] - mp_m) / mp_s, device=dev)[None]
            mm_ = torch.tensor(s['mmask'], device=dev)[None]
            logits, resid = model(ego, ag, am_, mp_, mm_, anchors)
            k = int(logits.argmax(-1))
        head_w = ego_to_world(densify(anchors_np[k], H), p0, h0)
        gt = sc.expert_future[:H]
        v = max(sc.speed_now(), 0.5)
        t = np.arange(1, H + 1) * 0.1
        cv = np.stack([p0[0] + v * t * np.cos(h0), p0[1] + v * t * np.sin(h0)], 1)
        rh = score_trajectory(sc, head_w)['driving_score']
        rg = score_trajectory(sc, gt)['driving_score']
        rc = score_trajectory(sc, cv)['driving_score']

        ax = axes[shown // 3, shown % 3]
        for r in (d.get('roads') or []):
            if len(r.get('geometry') or []) < 2:
                continue
            L = np.asarray([[q['x'], q['y']] for q in r['geometry']])
            c = '#d9d9d9' if r.get('type') == 'road_edge' else '#b0bec5'
            ax.plot(L[:, 0], L[:, 1], color=c, lw=0.8, zorder=1)
        for o in (d.get('objects') or [])[:40]:
            q = o.get('position') or []
            if len(q) <= sc.t_now:
                continue
            p = np.array([q[sc.t_now]['x'], q[sc.t_now]['y']])
            if np.linalg.norm(p - p0) > 60:
                continue
            hh = float((o.get('heading') or [0])[sc.t_now])
            w, ln = float(o.get('width') or 1.9), float(o.get('length') or 4.6)
            cs, sn = np.cos(hh), np.sin(hh)
            pts = np.array([[-ln / 2, -w / 2], [ln / 2, -w / 2],
                            [ln / 2, w / 2], [-ln / 2, w / 2]])
            pts = pts @ np.array([[cs, -sn], [sn, cs]]).T + p
            ax.add_patch(Polygon(pts, closed=True, color='#cfd8dc',
                                 ec='#90a4ae', lw=0.4, zorder=2))
        ax.plot(gt[:, 0], gt[:, 1], color='#4caf50', lw=2.4, label='expert', zorder=5)
        ax.plot(head_w[:, 0], head_w[:, 1], color='#e53935', lw=2.0,
                label='head', zorder=6)
        ax.plot(cv[:, 0], cv[:, 1], color='#9e9e9e', lw=1.4, ls='--',
                label='const-vel', zorder=4)
        ax.plot([p0[0]], [p0[1]], 'o', color='black', ms=4, zorder=7)
        ax.set_title(f'anchor {k} | PDMS: head {rh:.0f} / expert {rg:.0f} / const {rc:.0f}',
                     fontsize=9)
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        if shown == 0:
            ax.legend(fontsize=7, loc='best')
        shown += 1
    fig.suptitle('Trajectory comparison (green=expert, red=head, dashed=constant velocity)',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig3_trajectories.png')
    plt.close(fig)


def fig4():
    log = '/workspace/data/logs/train_medoid_dd.log'
    ep, fde, acc = [], [], []
    if os.path.exists(log):
        for line in open(log):
            m = re.match(r'ep\s+(\d+) loss .*p50 ([\d.]+).*acc ([\d.]+)%', line)
            if m:
                ep.append(int(m.group(1)))
                fde.append(float(m.group(2)))
                acc.append(float(m.group(3)))
    if not ep:
        return
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=160)
    ax.plot(ep, fde, 'o-', color='#e53935', ms=3, label='验证 FDE p50 (m)')
    ax.axhline(1.38, color='#e53935', ls=':', lw=1)
    ax.text(ep[-1], 1.38, ' anchor coverage limit 1.38', va='bottom', ha='right',
            fontsize=8, color='#e53935')
    ax.set_xlabel('epoch'); ax.set_ylabel('FDE p50 (m)', color='#e53935')
    ax.grid(alpha=0.25)
    ax2 = ax.twinx()
    ax2.plot(ep, acc, 's-', color='#2196f3', ms=3, label='anchor accuracy (%)')
    ax2.set_ylabel('anchor accuracy (%)', color='#2196f3')
    ax.set_title('Training curves (K=20 medoid, early stop ep110, best ep70)',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig4_training.png')
    plt.close(fig)


if __name__ == '__main__':
    fig1(); print('fig1 ok', flush=True)
    fig2(); print('fig2 ok', flush=True)
    fig3(); print('fig3 ok', flush=True)
    fig4(); print('fig4 ok', flush=True)
    print('ALL DONE ->', OUT)
