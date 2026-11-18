"""三种周车预测模式（cv / true / inflated）下的 G1 vs G2 归因实验。

一次遍历，记录每条候选在三种模式下的 plan/exec 判定；评测统一用真实未来。
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "tools"))

from metrics.openloop_pdms import (Scene, _min_dist, _obb_overlap,  # noqa: E402
                                   step_metrics)
from build_head_dataset import build_one  # noqa: E402
from eval_head_pdms import densify, ego_to_world  # noqa: E402
from exp_pre import pure_pursuit  # noqa: E402

D_EDGE = 1.0
HALF_W = 0.975
INFLATE = 0.3          # m/s：膨胀速率


def agents_for(sc, mode):
    """返回 (P, Hd, valid)：三种模式。"""
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
        d = sc.ap[a, t] - sc.ap[a, t - k]
        v = d / (k * 0.1)
        sp = np.linalg.norm(v)
        hd = float(np.arctan2(v[1], v[0])) if sp > 0.3 else float(sc.ah[a, t])
        for j in range(H):
            P[a, j] = p0 + v * (j + 1) * 0.1
            Hd[a, j] = hd
        V[a, :] = sc.av[a, t]
    return P, Hd, V


def check_traj(sc, traj, P, Hd, V, inflate=0.0):
    H = min(len(traj), sc.n_future)
    traj = traj[:H]
    ec = min((_min_dist(p, sc.ea, sc.eb) for p in traj), default=1e9) \
        if len(sc.ea) else 1e9
    ok_edge = ec >= D_EDGE
    coll = False
    for tt in range(H):
        p = traj[tt]
        dv = traj[tt + 1] - p if tt + 1 < H else p - traj[tt - 1]
        h = float(np.arctan2(dv[1], dv[0])) if np.linalg.norm(dv) > 1e-6 else 0.0
        dl = inflate * (tt + 1) * 0.1
        for a in range(len(P)):
            if not V[a, tt]:
                continue
            q = P[a, tt]
            if np.linalg.norm(q - p) > 6.0 + dl:
                continue
            da = (sc.ad[a][0] + 2 * dl, sc.ad[a][1] + 2 * dl)
            if _obb_overlap(p, h, sc.ego_dim, q, float(Hd[a, tt]), da):
                coll = True
                break
        if coll:
            break
    return (ok_edge and not coll), float(ec)


def exec_eval(sc, traj):
    H = min(len(traj), sc.n_future)
    traj = traj[:H]
    m = step_metrics(sc, traj)
    coll = bool(np.any(m['coll']))
    off = False
    if len(sc.ea):
        for p in traj:
            if _min_dist(p, sc.ea, sc.eb) < HALF_W + 0.1:
                off = True
                break
    return coll, off, float(sc.progress(traj[-1]))


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
    ap.add_argument('--out', default='/workspace/data/screen2')
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
    print(f'评测场景 {len(names)}', flush=True)

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
        p0, h0 = sc.expert_pos[sc.t_now], float(sc.expert_heading[sc.t_now])
        with torch.no_grad():
            ego = torch.tensor((s['ego'] - ego_m) / ego_s, device=dev)[None]
            ag = torch.tensor((s['agent'] - ag_m) / ag_s, device=dev)[None]
            am_ = torch.tensor(s['amask'], device=dev)[None]
            mp_ = torch.tensor((s['map'] - mp_m) / mp_s, device=dev)[None]
            mm_ = torch.tensor(s['mmask'], device=dev)[None]
            logits, _ = model(ego, ag, am_, mp_, mm_, anchors)
            prob = torch.softmax(logits, -1)[0].cpu().numpy()

        Mode = {m: agents_for(sc, m) for m in ('cv', 'true')}
        v0 = max(sc.speed_now(), 1.0)
        sid = len(rows)
        for r, k in enumerate(np.argsort(-prob)[:args.k_top]):
            plan = ego_to_world(densify(A[k], sc.n_future), p0, h0)
            try:
                roll = pure_pursuit(plan, v0)
            except Exception:
                roll = plan
            H = sc.n_future
            if len(roll) < H:
                roll = np.vstack([roll, np.repeat(roll[-1:], H - len(roll), 0)])
            roll = roll[:H]
            coll, off, prog = exec_eval(sc, roll)
            row = dict(scene=sid, rank=r, k=int(k), prob=float(prob[k]),
                       coll=bool(coll), off=bool(off), prog=float(prog))
            for m in ('cv', 'true'):
                P, Hd, V = Mode[m]
                ok_p, _ = check_traj(sc, plan, P, Hd, V)
                ok_e, _ = check_traj(sc, roll, P, Hd, V)
                row[f'okp_{m}'] = bool(ok_p)
                row[f'oke_{m}'] = bool(ok_e)
            P, Hd, V = Mode['cv']
            ok_p, _ = check_traj(sc, plan, P, Hd, V, inflate=INFLATE)
            ok_e, _ = check_traj(sc, roll, P, Hd, V, inflate=INFLATE)
            row['okp_inf'] = bool(ok_p)
            row['oke_inf'] = bool(ok_e)
            rows.append(row)
        if (i + 1) % 200 == 0:
            print(f'  {i+1}/{len(names)}', flush=True)

    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out, 'screen2.npz'),
                        rows=json.dumps(rows))

    # ---------- 分析 ----------
    by = {}
    for r in rows:
        by.setdefault(r['scene'], []).append(r)
    for s in by.values():
        s.sort(key=lambda x: x['rank'])
    n = len(by)

    def run_mode(mode):
        res = {}
        for g in ('G0', 'G1', 'G2'):
            res[g] = dict(coll=0, off=0, prog=[], allfail=0, changed=0)
        for sid, srows in by.items():
            g0 = srows[0]
            g1, ok1 = first_ok(srows, f'okp_{mode}')
            g2, ok2 = first_ok(srows, f'oke_{mode}')
            for g, r in (('G0', g0), ('G1', g1), ('G2', g2)):
                res[g]['coll'] += int(r['coll'])
                res[g]['off'] += int(r['off'])
                res[g]['prog'].append(r['prog'])
            if not ok1:
                res['G1']['allfail'] += 1
            if not ok2:
                res['G2']['allfail'] += 1
            if g2['rank'] != g0['rank']:
                res['G2']['changed'] += 1
        return res

    print(f'\n===== 场景级（n={n}）=====')
    out = {}
    for mode in ('cv', 'true', 'inf'):
        res = run_mode(mode)
        out[mode] = res
        print(f'\n--- 预测模式: {mode} ---')
        print(f"{'组':4s} {'碰撞':>6s} {'越界':>6s} {'进度':>7s} "
              f"{'全拒':>5s} {'改选':>5s}")
        for g in ('G0', 'G1', 'G2'):
            r = res[g]
            print(f'{g:4s} {r["coll"]:6d} {r["off"]:6d} '
                  f'{np.mean(r["prog"]):7.3f} {r["allfail"]:5d} '
                  f'{r["changed"]:5d}')

    # PPEF（候选级）按模式
    print(f'\n===== 候选级 PPEF（plan 过 / exec 失败，共 {len(rows)}）=====')
    for mode in ('cv', 'true', 'inf'):
        pe = sum(1 for r in rows
                 if r[f'okp_{mode}'] and (r['coll'] or r['off']))
        print(f'  {mode:5s}: {pe:6d} ({100*pe/len(rows):.1f}%)')

    # 图
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), dpi=160)
    labels = ['G0\ntop-1', 'G1\nplan', 'G2\nrollout']
    x = np.arange(3)
    for ax, key, title in ((axes[0], 'coll', 'collisions'),
                           (axes[1], 'off', 'off-road')):
        w = 0.26
        for i, mode in enumerate(('cv', 'true', 'inf')):
            vals = [out[mode][g][key] for g in ('G0', 'G1', 'G2')]
            ax.bar(x + (i - 1) * w, vals, w, label=mode)
            for j in range(3):
                ax.text(x[j] + (i - 1) * w, vals[j] + 0.5, str(vals[j]),
                        ha='center', fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(f'scenes (n={n})')
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(f'{args.out}/fig_screen2.png')
    plt.close(fig)
    print('\n图已保存', args.out)


if __name__ == '__main__':
    main()
