"""E3：级联门控实验（头的 top-k + 走廊证书）。

策略：
  top1     —— 头的 argmax（基线）
  cascade  —— 按概率顺序取第一条"通过证书"的；全不通过则取裕度最大的
  oracle   —— top-k 里 PDMS 最高的（上界，不可实现）
指标：PDMS 均值/p10/p50、绝对碰撞率、压线率、全军覆没率、改变选择的代价
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
from exp_e1e2 import certify, traj_metrics  # noqa: E402


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

    res = {k: [] for k in ('top1', 'cascade', 'oracle')}
    coll = {k: 0 for k in res}
    off = {k: 0 for k in res}
    n_all_fail = 0
    n_changed = 0
    delta = []
    n_used = 0

    for fp in files:
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        sc = Scene(d, horizon=40)
        s = build_one(d, A)
        if s is None or sc.n_future < 20:
            continue
        n_used += 1
        p0, h0 = sc.expert_pos[sc.t_now], float(sc.expert_heading[sc.t_now])
        with torch.no_grad():
            ego = torch.tensor((s['ego'] - ego_m) / ego_s, device=dev)[None]
            ag = torch.tensor((s['agent'] - ag_m) / ag_s, device=dev)[None]
            am_ = torch.tensor(s['amask'], device=dev)[None]
            mp_ = torch.tensor((s['map'] - mp_m) / mp_s, device=dev)[None]
            mm_ = torch.tensor(s['mmask'], device=dev)[None]
            logits, _ = model(ego, ag, am_, mp_, mm_, anchors)
            prob = torch.softmax(logits, -1)[0].cpu().numpy()
        order = np.argsort(-prob)[:args.k_top]

        cands = []
        for k in order:
            world = ego_to_world(densify(A[k], sc.n_future), p0, h0)
            feat = traj_metrics(sc, world)
            r = score_trajectory(sc, world)
            cands.append(dict(k=int(k), prob=float(prob[k]), world=world,
                              feat=feat, score=r.get('driving_score', 0.0),
                              ok=not certify(feat)))

        top1 = cands[0]
        passed = [c for c in cands if c['ok']]
        if passed:
            cas = passed[0]
        else:
            n_all_fail += 1
            cas = max(cands, key=lambda c: c['feat']['edge_clear'])
        ora = max(cands, key=lambda c: c['score'])
        if cas['k'] != top1['k']:
            n_changed += 1

        for name, c in (('top1', top1), ('cascade', cas), ('oracle', ora)):
            res[name].append(c['score'])
            coll[name] += int(c['feat']['viol'] and c['feat']['coll'])
            off[name] += int(c['feat']['off'])
        delta.append(cas['score'] - top1['score'])

    def summ(a):
        a = np.asarray(a)
        return a.mean(), np.percentile(a, 10), np.percentile(a, 50)

    print(f"\n场景 {n_used}  top-k={args.k_top}")
    print(f"全军覆没率（top-k 全不通过证书）: "
          f"{100*n_all_fail/max(n_used,1):.1f}%  ({n_all_fail}/{n_used})")
    print(f"门控改变选择的场景: {100*n_changed/max(n_used,1):.1f}%")
    print(f"\n{'策略':10s} {'PDMS均值':>9s} {'p10':>6s} {'p50':>6s} "
          f"{'绝对撞车':>8s} {'压线':>6s}")
    for name in ('top1', 'cascade', 'oracle'):
        m, p10, p50 = summ(res[name])
        print(f"{name:10s} {m:9.1f} {p10:6.1f} {p50:6.1f} "
              f"{coll[name]:8d} {off[name]:6d}")
    dlt = np.asarray(delta)
    print(f"\n门控带来的逐场景 ΔPDMS: 均值 {dlt.mean():+.2f}, "
          f"变好 {int((dlt > 0.5).sum())} 个, 变差 {int((dlt < -0.5).sum())} 个")

    os.makedirs(args.out, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), dpi=160)
    names = ['top-1', 'cascade', 'oracle@5']
    keys = ['top1', 'cascade', 'oracle']
    means = [summ(res[k])[0] for k in keys]
    p10s = [summ(res[k])[1] for k in keys]
    x = np.arange(3)
    axes[0].bar(x - 0.18, means, 0.34, label='mean', color='#2196f3')
    axes[0].bar(x + 0.18, p10s, 0.34, label='p10 (tail)', color='#e53935')
    for i in range(3):
        axes[0].text(x[i] - 0.18, means[i] + 1.2, f'{means[i]:.1f}',
                     ha='center', fontsize=9)
        axes[0].text(x[i] + 0.18, p10s[i] + 1.2, f'{p10s[i]:.1f}',
                     ha='center', fontsize=9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names)
    axes[0].set_ylabel('PDMS')
    axes[0].set_ylim(0, 100)
    axes[0].set_title(f'E3: cascade gate (k={args.k_top}, '
                      f'all-fail {100*n_all_fail/max(n_used,1):.1f}%)')
    axes[0].legend(fontsize=8)
    axes[0].grid(axis='y', alpha=0.25)

    axes[1].hist(dlt, bins=30, color='#7e57c2', alpha=0.8)
    axes[1].axvline(0, color='black', ls='--', lw=1)
    axes[1].set_xlabel('per-scene ΔPDMS (cascade - top1)')
    axes[1].set_ylabel('count')
    axes[1].set_title(f'E3: gating effect (better {int((dlt>0.5).sum())}, '
                      f'worse {int((dlt<-0.5).sum())})')
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(f'{args.out}/fig_e3_gate.png')
    plt.close(fig)
    print('\n图已保存', args.out)


if __name__ == '__main__':
    main()
