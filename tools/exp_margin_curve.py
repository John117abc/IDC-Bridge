"""裕度-违规率曲线 + 2000 场景验证（一次遍历，多阈值扫描）。

对每个场景取规划头 top-5 → 对每条做纯追踪推演 → 记录：
  执行误差 ε(整条最大偏差)、推演轨迹是否碰撞、路缘净空、执行后 PDMS
然后对 d ∈ [0.3, 2.0] 扫：
  门控 = 取第一条"无碰撞且净空 ≥ d"的；全失败保留 top-1
输出：
  ① PDMS 均值/p10 vs d
  ② 理论违规上界 P(ε > d) vs 实测违规率 vs d
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
from exp_pre import pure_pursuit  # noqa: E402
from exp_e5 import rollout_metrics  # noqa: E402

D_GRID = np.round(np.arange(0.3, 2.05, 0.1), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--n-scenes', type=int, default=2000)
    ap.add_argument('--k-top', type=int, default=5)
    ap.add_argument('--seed', type=int, default=31)
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
    print(f'场景 {len(files)}', flush=True)

    # 每个场景存：每候选 (coll, edge_clear, pdms, gap)
    data = []
    eps_top1 = []
    for i, fp in enumerate(files):
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
        v0 = max(sc.speed_now(), 1.0)
        cands = []
        for k in np.argsort(-prob)[:args.k_top]:
            plan = ego_to_world(densify(A[k], sc.n_future), p0, h0)
            try:
                roll = pure_pursuit(plan, v0)
            except Exception:
                roll = plan
            H = sc.n_future
            if len(roll) < H:
                roll = np.vstack([roll, np.repeat(roll[-1:], H - len(roll), 0)])
            roll = roll[:H]
            fm = rollout_metrics(sc, roll)
            r = score_trajectory(sc, roll)
            gap = float(np.max(np.linalg.norm(roll - plan[:H], axis=1)))
            cands.append(dict(coll=fm['coll'], edge=fm['edge_clear'],
                              pdms=r.get('driving_score', 0.0), gap=gap))
        data.append(cands)
        eps_top1.append(cands[0]['gap'])
        if (i + 1) % 200 == 0:
            print(f'  {i+1}/{len(files)}', flush=True)

    eps = np.asarray(eps_top1)
    print(f'\n执行误差 ε（top-1 整条最大偏差）: '
          f'p50={np.percentile(eps,50):.3f} p90={np.percentile(eps,90):.3f} '
          f'p99={np.percentile(eps,99):.3f}', flush=True)

    # top-1 基线
    base = np.asarray([c[0]['pdms'] for c in data])
    base_coll = sum(int(c[0]['coll']) for c in data)
    n = len(data)
    print(f'\n{"d":>5s} {"PDMS均值":>9s} {"p10":>6s} {"违规上界":>8s} '
          f'{"实测撞车":>8s} {"全失败":>7s}')
    rows = []
    for dd in D_GRID:
        sel_pdms, n_coll, n_fail = [], 0, 0
        for cands in data:
            ok = [((not c['coll']) and c['edge'] >= dd) for c in cands]
            if ok[0]:
                j = 0
            else:
                nxt = next((t for t, o in enumerate(ok) if o), None)
                if nxt is None:
                    j = 0
                    n_fail += 1
                else:
                    j = nxt
            sel_pdms.append(cands[j]['pdms'])
            n_coll += int(cands[j]['coll'])
        bound = float(np.mean(eps > dd))       # 理论违规概率上界
        rows.append((dd, np.mean(sel_pdms), np.percentile(sel_pdms, 10),
                     bound, n_coll / n, n_fail / n))
        print(f'{dd:5.2f} {np.mean(sel_pdms):9.1f} '
              f'{np.percentile(sel_pdms,10):6.1f} {100*bound:7.1f}% '
              f'{n_coll:8d} {100*n_fail/n:6.1f}%', flush=True)

    print(f'\n{"top-1":>5s} {base.mean():9.1f} {np.percentile(base,10):6.1f}')
    print(f'top-1 撞车 {base_coll}/{n}')

    os.makedirs(args.out, exist_ok=True)
    d = np.asarray([r[0] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), dpi=160)
    axes[0].plot(d, [r[1] for r in rows], 'o-', color='#2196f3', label='gated mean')
    axes[0].plot(d, [r[2] for r in rows], 's--', color='#e53935', label='gated p10')
    axes[0].axhline(base.mean(), color='gray', ls=':', label='top-1 mean')
    axes[0].axhline(np.percentile(base, 10), color='gray', ls='-.',
                    label='top-1 p10')
    axes[0].set_xlabel('certificate margin d (m)')
    axes[0].set_ylabel('executed PDMS')
    axes[0].set_title(f'(a) PDMS vs margin (n={n})')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    axes[1].plot(d, [100 * r[3] for r in rows], 'o-', color='#7e57c2',
                 label='bound: P(ε > d)')
    axes[1].plot(d, [100 * r[4] for r in rows], 's-', color='#e53935',
                 label='measured collision rate')
    axes[1].set_xlabel('certificate margin d (m)')
    axes[1].set_ylabel('%')
    axes[1].set_title('(b) theoretical bound vs measured')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(f'{args.out}/fig_margin_curve.png')
    plt.close(fig)
    print('图已保存', args.out)


if __name__ == '__main__':
    main()
