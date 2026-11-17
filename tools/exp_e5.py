"""E5：证书升级 —— 在"纯追踪推演轨迹"上做证书，并用矩形重叠查周车碰撞。

ok = (推演轨迹与周车无矩形重叠) and (推演轨迹距路缘 >= d_edge)
若 top-k 全不通过 → 保留 top-1（安全降级）
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

from metrics.openloop_pdms import (Scene, _min_dist, _obb_overlap,  # noqa: E402
                                   score_trajectory)
from build_head_dataset import build_one  # noqa: E402
from eval_head_pdms import densify, ego_to_world  # noqa: E402
from exp_pre import pure_pursuit  # noqa: E402

D_EDGE = 1.0
OFF_ABS = 0.5


def rollout_metrics(sc, traj):
    """在推演轨迹上算：距路缘最小净空、是否与周车矩形重叠、是否压线。"""
    H = min(len(traj), sc.n_future)
    traj = traj[:H]
    edge_clear = (min(_min_dist(p, sc.ea, sc.eb) for p in traj)
                  if len(sc.ea) else 1e9)
    off = any(_min_dist(p, sc.ea, sc.eb) < OFF_ABS for p in traj) \
        if len(sc.ea) else False
    coll = False
    n_coll_steps = 0
    for t in range(H):
        at = sc.f0 + t
        p = traj[t]
        d = np.linalg.norm(sc.ap[:, at, :] - p[None], axis=1) if len(sc.ap) else []
        if len(sc.ap):
            near = np.where((d < 6.0) & sc.av[:, at])[0]
            for a in near:
                if _obb_overlap(p, _heading(traj, t), sc.ego_dim,
                                sc.ap[a, at], sc.ah[a, at], sc.ad[a]):
                    coll = True
                    n_coll_steps += 1
                    break
    return dict(edge_clear=float(edge_clear), coll=coll,
                n_coll_steps=n_coll_steps, off=bool(off))


def _heading(traj, t):
    if t + 1 < len(traj):
        d = traj[t + 1] - traj[t]
    else:
        d = traj[t] - traj[t - 1]
    return float(np.arctan2(d[1], d[0])) if np.linalg.norm(d) > 1e-6 else 0.0


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

    res = {k: {'exec': [], 'gap': [], 'coll': 0, 'off': 0}
           for k in ('top1', 'gate')}
    rej_coll = rej_edge = 0
    all_fail = changed = n = 0

    for fp in files:
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        sc = Scene(d, horizon=40)
        s = build_one(d, A)
        if s is None or sc.n_future < 20:
            continue
        n += 1
        H = sc.n_future
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
            plan = ego_to_world(densify(A[k], H), p0, h0)
            try:
                roll = pure_pursuit(plan, v0)
            except Exception:
                roll = plan
            if len(roll) < H:
                roll = np.vstack([roll, np.repeat(roll[-1:], H - len(roll), 0)])
            roll = roll[:H]
            fm = rollout_metrics(sc, roll)
            r = score_trajectory(sc, roll)
            ok = (not fm['coll']) and fm['edge_clear'] >= D_EDGE
            cands.append(dict(plan=plan, roll=roll, fm=fm, ok=ok,
                              score=r.get('driving_score', 0.0)))
            if not ok:
                if fm['coll']:
                    rej_coll += 1
                else:
                    rej_edge += 1

        passed = [c for c in cands if c['ok']]
        if passed:
            gate = passed[0]
            if gate is not cands[0]:
                changed += 1
        else:
            gate = cands[0]
            all_fail += 1

        for name, c in (('top1', cands[0]), ('gate', gate)):
            res[name]['exec'].append(c['score'])
            res[name]['gap'].append(float(np.mean(
                np.linalg.norm(c['roll'] - c['plan'], axis=1))))
            res[name]['coll'] += int(c['fm']['coll'])
            res[name]['off'] += int(c['fm']['off'])

    def sm(a):
        a = np.asarray(a)
        return a.mean(), np.percentile(a, 10), np.percentile(a, 50)

    print(f"\n场景 {n}  全失败 {100*all_fail/max(n,1):.1f}%  "
          f"改变选择 {100*changed/max(n,1):.1f}%")
    print(f"候选被拒原因：碰撞 {rej_coll} 条，路缘 {rej_edge} 条")
    print(f"{'策略':6s} {'执行PDMS':>9s} {'p10':>6s} {'p50':>6s} "
          f"{'gapADE':>7s} {'撞车':>5s} {'压线':>5s}")
    for name, label in (('top1', 'top-1'), ('gate', 'E5gate')):
        em, e10, e50 = sm(res[name]['exec'])
        gm, _, _ = sm(res[name]['gap'])
        print(f"{label:6s} {em:9.1f} {e10:6.1f} {e50:6.1f} {gm:7.3f} "
              f"{res[name]['coll']:5d} {res[name]['off']:5d}")

    os.makedirs(args.out, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=160)
    labels = ['top-1', 'E5 gate']
    means = [sm(res['top1']['exec'])[0], sm(res['gate']['exec'])[0]]
    p10s = [sm(res['top1']['exec'])[1], sm(res['gate']['exec'])[1]]
    x = np.arange(2)
    ax.bar(x - 0.18, means, 0.34, label='executed PDMS mean', color='#2196f3')
    ax.bar(x + 0.18, p10s, 0.34, label='executed p10', color='#e53935')
    for i in range(2):
        ax.text(x[i] - 0.18, means[i] + 1, f'{means[i]:.1f}', ha='center',
                fontsize=9)
        ax.text(x[i] + 0.18, p10s[i] + 1, f'{p10s[i]:.1f}', ha='center',
                fontsize=9)
        ax.text(x[i], 3, f"coll {res['top1' if i==0 else 'gate']['coll']} / "
                          f"off {res['top1' if i==0 else 'gate']['off']}",
                ha='center', fontsize=8, color='white')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('executed PDMS')
    ax.set_ylim(0, 100)
    ax.set_title('E5: certificate on tracked rollout (OBB collision check)')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(f'{args.out}/fig_e5_rollout_cert.png')
    plt.close(fig)
    print('\n图已保存', args.out)


if __name__ == '__main__':
    main()
