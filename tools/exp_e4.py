"""E4：非反应式闭环 —— 规划轨迹用纯追踪真实执行，再评估执行后的轨迹。

策略：top-1（无门控） vs veto 级联（d_edge=1.0）
指标：执行后 PDMS、执行 gap（ADE/FDE）、执行后绝对碰撞/压线、
      以及"通过证书的规划在执行后仍然碰撞"的比例（C4 核心）
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
from exp_pre import pure_pursuit  # noqa: E402

D_EDGE = 1.0


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

    res = {k: {'plan': [], 'exec': [], 'gap': [], 'gapf': [],
               'coll': 0, 'off': 0} for k in ('top1', 'gate')}
    cert_collide = 0          # 通过证书、但执行后碰撞
    cert_total = 0
    changed = 0
    n = 0

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

        cands = []
        for k in np.argsort(-prob)[:args.k_top]:
            world = ego_to_world(densify(A[k], H), p0, h0)
            feat = traj_metrics(sc, world)
            ok = feat['edge_clear'] >= D_EDGE
            r = score_trajectory(sc, world)
            cands.append(dict(world=world, feat=feat, ok=ok,
                              plan=r.get('driving_score', 0.0)))
        passed = [c for c in cands if c['ok']]
        gate = passed[0] if passed else cands[0]
        if gate is not cands[0]:
            changed += 1
        # C4：被证书放行、且实际执行后是否碰撞
        if gate['ok']:
            cert_total += 1

        v0 = max(sc.speed_now(), 1.0)
        for name, c in (('top1', cands[0]), ('gate', gate)):
            try:
                exe = pure_pursuit(c['world'], v0)
            except Exception:
                continue
            if len(exe) < H:
                exe = np.vstack([exe, np.repeat(exe[-1:], H - len(exe), 0)])
            exe = exe[:H]
            feat_e = traj_metrics(sc, exe)
            r_e = score_trajectory(sc, exe)
            res[name]['plan'].append(c['plan'])
            res[name]['exec'].append(r_e.get('driving_score', 0.0))
            res[name]['gap'].append(float(np.mean(
                np.linalg.norm(exe - c['world'][:H], axis=1))))
            res[name]['gapf'].append(float(np.linalg.norm(
                exe[-1] - c['world'][H - 1])))
            res[name]['coll'] += int(feat_e['coll'])
            res[name]['off'] += int(feat_e['off'])
            if name == 'gate' and c['ok'] and feat_e['coll']:
                cert_collide += 1

    def sm(a):
        a = np.asarray(a)
        return a.mean(), np.percentile(a, 10), np.percentile(a, 50)

    print(f"\n场景 {n}  门控改变选择 {100*changed/max(n,1):.1f}%")
    print(f"{'策略':6s} {'规划PDMS':>9s} {'执行PDMS':>9s} {'p10':>6s} "
          f"{'gapADE':>7s} {'gapFDE':>7s} {'撞车':>5s} {'压线':>5s}")
    for name, label in (('top1', 'top-1'), ('gate', 'gate')):
        pm, _, _ = sm(res[name]['plan'])
        em, e10, e50 = sm(res[name]['exec'])
        gm, _, _ = sm(res[name]['gap'])
        gf, _, _ = sm(res[name]['gapf'])
        print(f"{label:6s} {pm:9.1f} {em:9.1f} {e10:6.1f} {gm:7.3f} "
              f"{gf:7.3f} {res[name]['coll']:5d} {res[name]['off']:5d}")
    g_all = np.asarray(res['gate']['gap'])
    print(f"\nC4：证书放行 {cert_total} 条，执行后仍碰撞 {cert_collide} 条 "
          f"({100*cert_collide/max(cert_total,1):.1f}%)")
    print(f"执行偏差 gap: p50={np.percentile(g_all,50):.3f} "
          f"p90={np.percentile(g_all,90):.3f} p99={np.percentile(g_all,99):.3f} m "
          f"（证书裕度 {D_EDGE} m）")

    os.makedirs(args.out, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), dpi=160)
    labels = ['top-1 plan', 'top-1 exec', 'gate plan', 'gate exec']
    vals = [sm(res['top1']['plan'])[0], sm(res['top1']['exec'])[0],
            sm(res['gate']['plan'])[0], sm(res['gate']['exec'])[0]]
    p10s = [sm(res['top1']['plan'])[1], sm(res['top1']['exec'])[1],
            sm(res['gate']['plan'])[1], sm(res['gate']['exec'])[1]]
    x = np.arange(4)
    axes[0].bar(x - 0.18, vals, 0.34, label='mean', color='#2196f3')
    axes[0].bar(x + 0.18, p10s, 0.34, label='p10', color='#e53935')
    for i in range(4):
        axes[0].text(x[i] - 0.18, vals[i] + 1, f'{vals[i]:.1f}',
                     ha='center', fontsize=8)
        axes[0].text(x[i] + 0.18, p10s[i] + 1, f'{p10s[i]:.1f}',
                     ha='center', fontsize=8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=8, rotation=15)
    axes[0].set_ylabel('PDMS')
    axes[0].set_ylim(0, 100)
    axes[0].set_title('E4: planned vs executed (kinematic bicycle)')
    axes[0].legend(fontsize=8)
    axes[0].grid(axis='y', alpha=0.25)

    axes[1].hist(res['gate']['gap'], bins=30, color='#7e57c2', alpha=0.85)
    axes[1].axvline(D_EDGE, color='#e53935', ls='--',
                    label=f'certificate margin {D_EDGE} m')
    axes[1].set_xlabel('execution gap ADE (m)')
    axes[1].set_ylabel('count')
    axes[1].set_title('E4: plan-execution gap vs certificate margin')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(f'{args.out}/fig_e4_closedloop.png')
    plt.close(fig)
    print('\n图已保存', args.out)


if __name__ == '__main__':
    main()
