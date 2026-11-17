"""E1 + E2：失效普查 + 证书当分类器。

E1：头 top-1 / top-3 / top-5 候选里，有多少条**自身不可行**（撞/压线），失效类型分布
E2：把走廊证书当二分类器，对"实际是否违规"做 precision/recall
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
                                   step_metrics)
from build_head_dataset import build_one  # noqa: E402
from eval_head_pdms import densify, ego_to_world  # noqa: E402

D_SAFE_EDGE = 1.5      # 证书：距路缘最小净空
D_SAFE_AGENT = 0.5     # 证书：距周车最小净空
KAPPA_MAX = 0.20       # 1/m
ALAT_MAX = 3.5         # m/s^2
OFF_ABS = 0.5          # 真值：中心距路缘 < 0.5 m 视为压线


def ego_radius(dim):
    return 0.5 * float(np.hypot(dim[0], dim[1]))


def traj_metrics(sc, traj):
    """真值 + 证书特征。"""
    H = min(len(traj), sc.n_future)
    traj = traj[:H]
    m = step_metrics(sc, traj)
    coll = bool(np.any(m['coll']))
    # 绝对压线（不经过人类过滤）
    off = False
    for p in traj:
        if len(sc.ea) and _min_dist(p, sc.ea, sc.eb) < OFF_ABS:
            off = True
            break
    # 证书特征
    edge_clear = min((_min_dist(p, sc.ea, sc.eb) for p in traj),
                     default=1e9) if len(sc.ea) else 1e9
    eg_r = ego_radius(sc.ego_dim)
    agent_clear = 1e9
    for t in range(H):
        at = sc.f0 + t
        for a in range(len(sc.ap)):
            if not sc.av[a, at]:
                continue
            d = float(np.linalg.norm(sc.ap[a, at] - traj[t]))
            agent_clear = min(agent_clear, d - eg_r - ego_radius(sc.ad[a]))
    d = np.diff(traj, axis=0)
    sp = np.linalg.norm(d, axis=1) / 0.1
    hd = np.arctan2(d[:, 1], d[:, 0])
    dh = np.arctan2(np.sin(np.diff(hd)), np.cos(np.diff(hd)))
    ds = np.maximum(sp[:-1], 0.1) * 0.1
    kappa = np.abs(dh) / np.maximum(ds, 0.1) if len(dh) else np.zeros(1)
    alat = float(np.max(sp[:-1] ** 2 * kappa)) if len(dh) else 0.0
    return dict(viol=coll or off, coll=coll, off=off,
                edge_clear=float(edge_clear), agent_clear=float(agent_clear),
                kappa=float(np.max(kappa)) if len(kappa) else 0.0,
                alat=alat)


def certify(feat):
    return (feat['edge_clear'] < D_SAFE_EDGE or
            feat['agent_clear'] < D_SAFE_AGENT or
            feat['kappa'] > KAPPA_MAX or feat['alat'] > ALAT_MAX)


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

    # top-1 / top-3 / top-5 的不可行率
    stat = {k: {'n': 0, 'viol': 0, 'coll': 0, 'off': 0, 'both': 0}
            for k in (1, 3, 5)}
    # E2 样本：(cert_score 越小越危险, 是否真的违规)
    cert_neg, cert_pos = [], []      # 真值违规 / 安全 的证书裕度
    TP = FP = FN = TN = 0
    scenes_used = 0

    for fp in files:
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        sc = Scene(d, horizon=40)
        s = build_one(d, A)
        if s is None or sc.n_future < 20:
            continue
        scenes_used += 1
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

        feats = []
        for k in order:
            world = ego_to_world(densify(A[k], sc.n_future), p0, h0)
            feats.append(traj_metrics(sc, world))

        for m_ in (1, 3, 5):
            if len(feats) < m_:
                continue
            # "选中的那条"按概率取第 m_ 个（模拟不同候选预算下的实际选择）
            f = feats[m_ - 1]
            st = stat[m_]
            st['n'] += 1
            if f['viol']:
                st['viol'] += 1
                st['coll'] += int(f['coll'])
                st['off'] += int(f['off'])
                st['both'] += int(f['coll'] and f['off'])

        for f in feats:
            margin = min(f['edge_clear'] - D_SAFE_EDGE,
                         f['agent_clear'] - D_SAFE_AGENT,
                         (KAPPA_MAX - f['kappa']) * 10.0,
                         (ALAT_MAX - f['alat']))
            pred = margin < 0
            if f['viol']:
                cert_pos.append(-margin)
            else:
                cert_neg.append(-margin)
            if pred and f['viol']:
                TP += 1
            elif pred and not f['viol']:
                FP += 1
            elif (not pred) and f['viol']:
                FN += 1
            else:
                TN += 1

    prec = TP / max(TP + FP, 1)
    rec = TP / max(TP + FN, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    print(f"\n场景 {scenes_used}，候选 {TP+FP+FN+TN} 条")
    print('\n=== E1 失效普查（按概率选第 m 个候选）===')
    print(f"{'策略':10s} {'样本':>5s} {'不可行率':>8s} {'纯撞车':>7s} "
          f"{'纯压线':>7s} {'两者':>6s}")
    for m_ in (1, 3, 5):
        st = stat[m_]
        if st['n']:
            print(f"top{m_:<7d} {st['n']:5d} {100*st['viol']/st['n']:7.1f}% "
                  f"{st['coll']:7d} {st['off']:7d} {st['both']:6d}")
    print('\n=== E2 证书当分类器 ===')
    print(f"TP={TP} FP={FP} FN={FN} TN={TN}")
    print(f"precision={prec*100:.1f}%  recall={rec*100:.1f}%  F1={f1*100:.1f}%")

    os.makedirs(args.out, exist_ok=True)
    # ---- 图 1：失效类型 ----
    fig, ax = plt.subplots(figsize=(8, 4), dpi=160)
    labels = ['top-1', 'top-3', 'top-5']
    viol = [100 * stat[m_]['viol'] / max(stat[m_]['n'], 1) for m_ in (1, 3, 5)]
    coll = [100 * stat[m_]['coll'] / max(stat[m_]['n'], 1) for m_ in (1, 3, 5)]
    off = [100 * stat[m_]['off'] / max(stat[m_]['n'], 1) for m_ in (1, 3, 5)]
    x = np.arange(3)
    ax.bar(x - 0.22, coll, 0.2, label='collision only/with', color='#e53935')
    ax.bar(x, off, 0.2, label='off-road', color='#fb8c00')
    ax.bar(x + 0.22, viol, 0.2, label='any infeasible', color='#546e7a')
    for i, v in enumerate(viol):
        ax.text(x[i] + 0.22, v + 0.5, f'{v:.1f}%', ha='center', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('% of scenes')
    ax.set_title(f'E1: infeasible-plan rate (n={scenes_used} scenes)')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(f'{args.out}/fig_e1_failure.png')
    plt.close(fig)

    # ---- 图 2：证书裕度分布 + 阈值扫描 ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=160)
    axes[0].hist(cert_neg, bins=40, alpha=0.65, color='#43a047', label='safe')
    axes[0].hist(cert_pos, bins=40, alpha=0.65, color='#e53935', label='violating')
    axes[0].axvline(0, color='black', ls='--', lw=1)
    axes[0].set_xlabel('certificate danger score (-margin, higher = worse)')
    axes[0].set_ylabel('count')
    axes[0].set_title('E2: certificate danger score distribution')
    axes[0].legend(fontsize=8)
    axes[0].set_xlim(-8, 8)
    # 阈值扫描
    all_m = np.array(cert_neg + cert_pos)
    all_y = np.r_[np.zeros(len(cert_neg)), np.ones(len(cert_pos))]
    ths = np.linspace(-5, 5, 60)
    P, R = [], []
    for th in ths:
        pred = all_m >= th
        p = (pred & (all_y == 1)).sum() / max(pred.sum(), 1)
        r = (pred & (all_y == 1)).sum() / max((all_y == 1).sum(), 1)
        P.append(p); R.append(r)
    axes[1].plot(ths, P, label='precision')
    axes[1].plot(ths, R, label='recall')
    axes[1].axvline(0, color='black', ls='--', lw=1)
    axes[1].set_xlabel('danger threshold (flag if danger >= thr)')
    axes[1].set_ylabel('score')
    axes[1].set_title(f'E2: P/R = {prec*100:.0f}% / {rec*100:.0f}% @ thr=0')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(f'{args.out}/fig_e2_certificate.png')
    plt.close(fig)
    print('\n图已保存到', args.out)


if __name__ == '__main__':
    main()
