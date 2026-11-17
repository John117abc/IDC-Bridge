"""E3-v3：二维扫描（路缘阈值 × 周车阈值），一次遍历所有组合。

输出：最佳组合、PDMS 均值/p10 热图、全军覆没率热图。
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "tools"))

from metrics.openloop_pdms import Scene, score_trajectory  # noqa: E402
from build_head_dataset import build_one  # noqa: E402
from eval_head_pdms import densify, ego_to_world  # noqa: E402
from exp_e1e2 import traj_metrics  # noqa: E402

EDGES = [0.6, 0.8, 1.0, 1.2]
AGENTS = [0.0, 0.1, 0.2, 0.3, 0.5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--n-scenes', type=int, default=150)
    ap.add_argument('--k-top', type=int, default=5)
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--out', default='/workspace/data/report')
    args = ap.parse_args()

    import torch
    from train_head import Head
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    A = np.load(os.path.join(args.data, 'anchors.npy'))
    K, T = A.shape[0], A.shape[1]
    ck = torch.load(args.model, map_location='cpu')
    model = Head(ck.get('K', K), ck.get('T', T))
    model.load_state_dict(ck['model'])
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(dev).eval()
    anchors = torch.tensor(A, dtype=torch.float32, device=dev)

    z = np.load(os.path.join(args.data, 'train.npz'))
    ego_m, ego_s = z['ego'].mean(0), z['ego'].std(0) + 1e-3
    am = z['amask'].astype(bool)
    ag_m, ag_s = z['agent'][am].mean(0), z['agent'][am].std(0) + 1e-3
    mm = z['mmask'].astype(bool)
    mp_m, mp_s = z['map'][mm].mean(0), z['map'][mm].std(0) + 1e-3

    files = sorted(glob.glob(os.path.join(args.data_dir, 'tfrecord-*.json')))
    rng = np.random.default_rng(args.seed)
    files = [files[i] for i in rng.choice(len(files), args.n_scenes,
                                          replace=False)]

    per_scene = []
    for fp in files:
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        sc = Scene(d, horizon=40)
        s = build_one(d, A)
        if s is None or sc.n_future < 20:
            continue
        p0, h0 = sc.expert_pos[sc.t_now], float(sc.expert_heading[sc.t_now])
        with torch.no_grad():
            ego = torch.tensor((s['ego'] - ego_m) / ego_s, device=dev)[None]
            ag = torch.tensor((s['agent'] - ag_m) / ag_s, device=dev)[None]
            am_ = torch.tensor(s['amask'], device=dev)[None]
            mp_ = torch.tensor((s['map'] - mp_m) / mp_s, device=dev)[None]
            mm_ = torch.tensor(s['mmask'], device=dev)[None]
            logits, _ = model(ego, ag, am_, mp_, mm_, anchors)
            prob = torch.softmax(logits, -1)[0].cpu().numpy()
        cands = []
        for k in np.argsort(-prob)[:args.k_top]:
            world = ego_to_world(densify(A[k], sc.n_future), p0, h0)
            feat = traj_metrics(sc, world)
            r = score_trajectory(sc, world)
            cands.append(dict(feat=feat, score=r.get('driving_score', 0.0)))
        per_scene.append(cands)

    n = len(per_scene)
    top1 = np.asarray([c[0]['score'] for c in per_scene])
    ora = np.asarray([max(x['score'] for x in c) for c in per_scene])
    print(f"场景 {n}")
    print(f"top-1 : mean {top1.mean():.1f}  p10 {np.percentile(top1,10):.1f}")
    print(f"oracle: mean {ora.mean():.1f}  p10 {np.percentile(ora,10):.1f}")

    mean_grid = np.zeros((len(EDGES), len(AGENTS)))
    p10_grid = np.zeros_like(mean_grid)
    fail_grid = np.zeros_like(mean_grid)
    coll_grid = np.zeros_like(mean_grid)
    best = (-1, None)
    for i, de in enumerate(EDGES):
        for j, da in enumerate(AGENTS):
            pdms, coll, off, fail = [], 0, 0, 0
            for cands in per_scene:
                ok = [c['feat']['edge_clear'] >= de and
                      c['feat']['agent_clear'] >= da for c in cands]
                sel = 0
                if not ok[0]:
                    nxt = next((t for t, o in enumerate(ok) if o), None)
                    if nxt is None:
                        fail += 1
                    else:
                        sel = nxt
                c = cands[sel]
                pdms.append(c['score'])
                coll += int(c['feat']['coll'])
                off += int(c['feat']['off'])
            p = np.asarray(pdms)
            mean_grid[i, j] = p.mean()
            p10_grid[i, j] = np.percentile(p, 10)
            fail_grid[i, j] = 100 * fail / n
            coll_grid[i, j] = coll
            if p.mean() > best[0]:
                best = (p.mean(), (de, da, p.mean(), np.percentile(p, 10),
                                   coll, off, 100 * fail / n))

    print(f"\n最佳: d_edge={best[1][0]}  d_agent={best[1][1]}  "
          f"mean={best[1][2]:.1f}  p10={best[1][3]:.1f}  "
          f"撞车={best[1][4]}  压线={best[1][5]}  全失败={best[1][6]:.1f}%")

    print(f"\n{'d_edge':>7s} " + " ".join(f"a={a:<5.1f}" for a in AGENTS))
    for i, de in enumerate(EDGES):
        print(f"{de:7.1f} " + " ".join(f"{mean_grid[i,j]:6.1f}"
                                       for j in range(len(AGENTS))))

    os.makedirs(args.out, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), dpi=160)
    for ax, grid, title, cmap, fmt in (
            (axes[0], mean_grid, 'PDMS mean', 'viridis', '{:.1f}'),
            (axes[1], fail_grid, 'all-fail %', 'magma_r', '{:.0f}')):
        im = ax.imshow(grid, aspect='auto', cmap=cmap, origin='lower')
        ax.set_xticks(range(len(AGENTS)))
        ax.set_xticklabels([f'{a:.1f}' for a in AGENTS])
        ax.set_yticks(range(len(EDGES)))
        ax.set_yticklabels([f'{e:.1f}' for e in EDGES])
        ax.set_xlabel('agent clearance threshold (m)')
        ax.set_ylabel('road-edge clearance threshold (m)')
        ax.set_title(title)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                ax.text(j, i, fmt.format(grid[i, j]), ha='center',
                        va='center', fontsize=8,
                        color='white' if grid[i, j] < np.median(grid) else 'black')
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f'E3-v3: 2D threshold sweep (n={n} scenes, top-1 mean '
                 f'{top1.mean():.1f})', fontsize=11)
    fig.tight_layout()
    fig.savefig(f'{args.out}/fig_e3v3_grid.png')
    plt.close(fig)
    print('\n图已保存', args.out)


if __name__ == '__main__':
    main()
