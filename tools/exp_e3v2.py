"""E3-v2：重标定 + 只否决不排序 + 全失败保留 top-1。

扫描路缘阈值 d_edge，观察：全军覆没率 / 改变率 / PDMS / 撞车 / 压线。
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

EDGE_GRID = [0.4, 0.6, 0.8, 1.0, 1.2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--n-scenes', type=int, default=150)
    ap.add_argument('--k-top', type=int, default=5)
    ap.add_argument('--d-agent', type=float, default=0.3)
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

    # 每个场景先算好候选，再对所有阈值扫描
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
    print(f"场景 {n}  top-k={args.k_top}  d_agent={args.d_agent}")

    rows = []
    for d_edge in EDGE_GRID:
        pdms, coll, off = [], 0, 0
        all_fail = 0
        changed = 0
        for cands in per_scene:
            ok = [c['feat']['edge_clear'] >= d_edge and
                  c['feat']['agent_clear'] >= args.d_agent for c in cands]
            sel = 0
            if not ok[0]:
                nxt = next((i for i, o in enumerate(ok) if o), None)
                if nxt is None:
                    all_fail += 1          # 全失败 → 保留 top-1
                else:
                    sel = nxt
                    changed += 1
            c = cands[sel]
            pdms.append(c['score'])
            coll += int(c['feat']['coll'])
            off += int(c['feat']['off'])
        p = np.asarray(pdms)
        rows.append(dict(d_edge=d_edge, mean=p.mean(),
                         p10=np.percentile(p, 10),
                         p50=np.percentile(p, 50),
                         coll=coll, off=off, all_fail=all_fail,
                         changed=changed))

    top1 = np.asarray([c[0]['score'] for c in per_scene])
    ora = np.asarray([max(x['score'] for x in c) for c in per_scene])
    coll1 = sum(int(c[0]['feat']['coll']) for c in per_scene)
    off1 = sum(int(c[0]['feat']['off']) for c in per_scene)

    print(f"\n{'d_edge':>7s} {'全失败':>7s} {'改变':>6s} {'均值':>7s} "
          f"{'p10':>6s} {'p50':>6s} {'撞车':>5s} {'压线':>5s}")
    print(f"{'top-1':>7s} {'-':>7s} {'-':>6s} {top1.mean():7.1f} "
          f"{np.percentile(top1,10):6.1f} {np.percentile(top1,50):6.1f} "
          f"{coll1:5d} {off1:5d}")
    for r in rows:
        print(f"{r['d_edge']:7.2f} {100*r['all_fail']/n:6.1f}% "
              f"{100*r['changed']/n:5.1f}% {r['mean']:7.1f} {r['p10']:6.1f} "
              f"{r['p50']:6.1f} {r['coll']:5d} {r['off']:5d}")
    print(f"{'oracle':>7s} {'-':>7s} {'-':>6s} {ora.mean():7.1f} "
          f"{np.percentile(ora,10):6.1f} {np.percentile(ora,50):6.1f}")

    best = max(rows, key=lambda r: r['mean'])
    os.makedirs(args.out, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), dpi=160)
    xs = [r['d_edge'] for r in rows]
    axes[0].plot(xs, [r['mean'] for r in rows], 'o-', label='veto cascade')
    axes[0].axhline(top1.mean(), color='gray', ls='--', label='top-1 (no gate)')
    axes[0].axhline(ora.mean(), color='#2196f3', ls=':', label='oracle@5')
    axes[0].set_xlabel('road-edge clearance threshold (m)')
    axes[0].set_ylabel('PDMS mean')
    axes[0].set_title('E3-v2: veto cascade vs threshold')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    axes[1].plot(xs, [100 * r['all_fail'] / n for r in rows], 's-',
                 color='#e53935', label='all-fail %')
    axes[1].plot(xs, [100 * r['changed'] / n for r in rows], '^-',
                 color='#7e57c2', label='selection changed %')
    axes[1].plot(xs, [r['coll'] for r in rows], 'o-', color='#fb8c00',
                 label='collisions')
    axes[1].set_xlabel('road-edge clearance threshold (m)')
    axes[1].set_ylabel('% / count')
    axes[1].set_title('E3-v2: gate behaviour')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(f'{args.out}/fig_e3v2_gate.png')
    plt.close(fig)
    print('\n图已保存', args.out, ' 最佳阈值', best['d_edge'])


if __name__ == '__main__':
    main()
