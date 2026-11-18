"""Round-2 主实验：2×2 网格（plan/rollout × cv/true），判据对齐 + 配对分析。

判据统一：路缘净空阈值 ROAD_MARGIN = 1.075 m（= 半车宽 0.975 + 0.1），
筛选与评测使用同一个阈值、同一套 OBB 尺寸。
推演统一起点：所有候选从决策帧的观测位姿 (x, y, heading) 与同一 v0 起步。
"""
import argparse
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "tools"))

from metrics.openloop_pdms import Scene, _min_dist, _obb_overlap, step_metrics  # noqa: E402
from build_head_dataset import build_one  # noqa: E402
from eval_head_pdms import densify, ego_to_world  # noqa: E402
from exp_pre import pure_pursuit  # noqa: E402

ROAD_MARGIN = 1.075        # 筛选与评测共用


def agents_view(sc, mode):
    t = sc.t_now
    H = sc.n_future
    A = len(sc.ap)
    P = np.zeros((A, H, 2), np.float32)
    Hd = np.zeros((A, H), np.float32)
    V = np.ones((A, H), bool)
    if mode == 'true':
        for a in range(A):
            P[a] = sc.ap[a, t + 1:t + 1 + H]
            Hd[a] = sc.ah[a, t + 1:t + 1 + H]
            V[a] = sc.av[a, t + 1:t + 1 + H]
        return P, Hd, V
    k = min(3, t)
    for a in range(A):
        p0 = sc.ap[a, t]
        v = (sc.ap[a, t] - sc.ap[a, t - k]) / (k * 0.1)
        sp = np.linalg.norm(v)
        hd = float(np.arctan2(v[1], v[0])) if sp > 0.3 else float(sc.ah[a, t])
        for j in range(H):
            P[a, j] = p0 + v * (j + 1) * 0.1
            Hd[a, j] = hd
        V[a, :] = sc.av[a, t]
    return P, Hd, V


def judge(sc, traj, P, Hd, V):
    """返回 (ok, edge_clear, coll)。判据与评测完全一致。"""
    H = min(len(traj), sc.n_future)
    traj = traj[:H]
    ec = min((_min_dist(p, sc.ea, sc.eb) for p in traj), default=1e9) \
        if len(sc.ea) else 1e9
    ok_edge = ec >= ROAD_MARGIN
    coll = False
    for tt in range(H):
        p = traj[tt]
        dv = traj[tt + 1] - p if tt + 1 < H else p - traj[tt - 1]
        h = float(np.arctan2(dv[1], dv[0])) if np.linalg.norm(dv) > 1e-6 else 0.0
        for a in range(len(P)):
            if not V[a, tt]:
                continue
            q = P[a, tt]
            if np.linalg.norm(q - p) > 6.0:
                continue
            if _obb_overlap(p, h, sc.ego_dim, q, float(Hd[a, tt]), sc.ad[a]):
                coll = True
                break
        if coll:
            break
    return (ok_edge and not coll), float(ec), coll


def first_ok(srows, key):
    for r in srows:
        if r[key]:
            return r, True
    return srows[0], False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--files-json', required=True)
    ap.add_argument('--n-scenes', type=int, default=2000)
    ap.add_argument('--k-top', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='/workspace/data/grid2')
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

    names = json.load(open(args.files_json))
    rng = np.random.default_rng(args.seed)
    if len(names) > args.n_scenes:
        names = [names[i] for i in rng.choice(len(names), args.n_scenes,
                                              replace=False)]
    print(f'场景 {len(names)}  ROAD_MARGIN={ROAD_MARGIN}', flush=True)

    rows = []
    for i, n in enumerate(names):
        fp = os.path.join(args.data_dir, n)
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        sc = Scene(d, horizon=40)
        s = build_one(d, A)
        if s is None or sc.n_future < 20:
            continue
        t = sc.t_now
        p_dec = sc.expert_pos[t]
        h_dec = float(sc.expert_heading[t])
        with torch.no_grad():
            ego = torch.tensor((s['ego'] - ego_m) / ego_s, device=dev)[None]
            ag = torch.tensor((s['agent'] - ag_m) / ag_s, device=dev)[None]
            am_ = torch.tensor(s['amask'], device=dev)[None]
            mp_ = torch.tensor((s['map'] - mp_m) / mp_s, device=dev)[None]
            mm_ = torch.tensor(s['mmask'], device=dev)[None]
            logits, _ = model(ego, ag, am_, mp_, mm_, anchors)
            prob = torch.softmax(logits, -1)[0].cpu().numpy()
        CV = agents_view(sc, 'cv')
        TR = agents_view(sc, 'true')
        v0 = max(sc.speed_now(), 1.0)
        sid = len(rows)
        for r, k in enumerate(np.argsort(-prob)[:args.k_top]):
            plan = ego_to_world(densify(A[k], sc.n_future), p_dec, h_dec)
            try:
                roll = pure_pursuit(plan, v0, p0=p_dec, h0=h_dec)
            except Exception:
                roll = plan
            H = sc.n_future
            if len(roll) < H:
                roll = np.vstack([roll, np.repeat(roll[-1:], H - len(roll), 0)])
            roll = roll[:H]
            okp_cv, _, _ = judge(sc, plan, *CV)
            okp_tr, _, _ = judge(sc, plan, *TR)
            oke_cv, _, _ = judge(sc, roll, *CV)
            oke_tr, coll, _ = judge(sc, roll, *TR)
            # 评测（与 oke_tr 同判据；off 单独算）
            off = False
            if len(sc.ea):
                for p in roll:
                    if _min_dist(p, sc.ea, sc.eb) < ROAD_MARGIN:
                        off = True
                        break
            rows.append(dict(scene=sid, rank=r, k=int(k), prob=float(prob[k]),
                             okp_cv=bool(okp_cv), okp_tr=bool(okp_tr),
                             oke_cv=bool(oke_cv), oke_tr=bool(oke_tr),
                             coll=bool(coll), off=bool(off),
                             prog=float(sc.progress(roll[-1]))))
        if (i + 1) % 200 == 0:
            print(f'  {i+1}/{len(names)}', flush=True)

    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out, 'grid2.npz'),
                        rows=json.dumps(rows))

    # ---------------- 分析 ----------------
    by = {}
    for r in rows:
        by.setdefault(r['scene'], []).append(r)
    for v in by.values():
        v.sort(key=lambda x: x['rank'])
    n = len(by)

    POL = [('G0', None), ('G1cv', 'okp_cv'), ('G1true', 'okp_tr'),
           ('G2cv', 'oke_cv'), ('G2true', 'oke_tr')]

    def sel(mode):
        out = {}
        for sid, srows in by.items():
            if mode is None:
                r = srows[0]
            else:
                r, _ = first_ok(srows, mode)
            out[sid] = r
        return out

    S = {name: sel(key) for name, key in POL}
    base = S['G0']

    print(f'\n===== 场景级（n={n}）=====')
    print(f"{'策略':8s} {'碰撞':>6s} {'越界':>6s} {'全拒':>6s} "
          f"{'救回':>6s} {'新增':>6s} {'改选':>6s} {'进度':>7s}")
    summary = {}
    for name, key in POL:
        sel_r = S[name]
        coll = sum(int(r['coll']) for r in sel_r.values())
        off = sum(int(r['off']) for r in sel_r.values())
        fail = 0 if key is None else sum(
            1 for srows in by.values() if not any(x[key] for x in srows))
        rescue = new = changed = 0
        for sid, r in sel_r.items():
            b = base[sid]
            b_bad = b['coll'] or b['off']
            r_bad = r['coll'] or r['off']
            if b_bad and not r_bad:
                rescue += 1
            if not b_bad and r_bad:
                new += 1
            if r['rank'] != b['rank']:
                changed += 1
        prog = np.mean([r['prog'] for r in sel_r.values()])
        summary[name] = dict(coll=coll, off=off, fail=fail, rescue=rescue,
                             new=new, changed=changed, prog=prog)
        print(f'{name:8s} {coll:6d} {off:6d} {fail:6d} {rescue:6d} '
              f'{new:6d} {changed:6d} {prog:7.3f}')

    # 判据翻转（候选级）
    def flips(a, b):
        up = sum(1 for r in rows if not r[a] and r[b])
        dn = sum(1 for r in rows if r[a] and not r[b])
        return up, dn

    print(f'\n===== 候选级判据翻转（共 {len(rows)}）=====')
    pairs = [('预测效应 plan', 'okp_cv', 'okp_tr'),
             ('预测效应 rollout', 'oke_cv', 'oke_tr'),
             ('层级效应 cv', 'okp_cv', 'oke_cv'),
             ('层级效应 true', 'okp_tr', 'oke_tr')]
    for label, a, b in pairs:
        up, dn = flips(a, b)
        print(f'{label:18s}: 由拒转过 {up:5d} | 由过转拒 {dn:5d} '
              f'(净 {up-dn:+d})')

    print(f'\n===== 选择一致性（相对 G0 选中候选不同的场景数）=====')
    for name, _ in POL[1:]:
        diff = sum(1 for sid in by if S[name][sid]['rank'] != base[sid]['rank'])
        print(f'{name:8s}: {diff}/{n} ({100*diff/n:.1f}%)')

    # 图
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), dpi=160)
    names5 = [p[0] for p in POL]
    x = np.arange(len(names5))
    axes[0].bar(x, [summary[p]['coll'] for p in names5], 0.6, color='#e53935')
    for i, p in enumerate(names5):
        axes[0].text(x[i], summary[p]['coll'] + 0.8, str(summary[p]['coll']),
                     ha='center', fontsize=9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names5, fontsize=8, rotation=15)
    axes[0].set_ylabel(f'collisions (n={n})')
    axes[0].set_title('Collisions by policy (aligned criteria)')
    axes[0].grid(axis='y', alpha=0.25)
    axes[1].bar(x, [summary[p]['off'] for p in names5], 0.6, color='#fb8c00')
    for i, p in enumerate(names5):
        axes[1].text(x[i], summary[p]['off'] + 0.8, str(summary[p]['off']),
                     ha='center', fontsize=9)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names5, fontsize=8, rotation=15)
    axes[1].set_ylabel(f'off-road (n={n})')
    axes[1].set_title('Off-road by policy')
    axes[1].grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(f'{args.out}/fig_grid2.png')
    plt.close(fig)
    print('\n图已保存', args.out)


if __name__ == '__main__':
    main()
