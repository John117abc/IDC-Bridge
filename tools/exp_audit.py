"""审计实验：两种选择策略 × 两种信息条件（修正一帧错位）。

策略：
  S1 = 首个通过即选（现状）
  S2 = PDM 式推演评分（对所有候选算加权分，取最高）
信息条件：
  cv   = 恒速预测（可部署）
  true = 真实未来（特权，仅诊断）
判据：ROAD_MARGIN=1.075，OBB 同尺寸；时间严格对齐（计划从 t+1 起，推演从 t 起）
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

from metrics.openloop_pdms import Scene, _min_dist, _obb_overlap  # noqa: E402
from build_head_dataset import build_one  # noqa: E402
from eval_head_pdms import densify, ego_to_world  # noqa: E402
from exp_pre import pure_pursuit  # noqa: E402

ROAD_MARGIN = 1.075


def agents_view(sc, mode):
    """返回长度 H+1 的数组：索引 j 对应该车在 t+j 的位置（j=0 为当前帧）。"""
    t = sc.t_now
    H = sc.n_future
    A = len(sc.ap)
    P = np.zeros((A, H + 1, 2), np.float32)
    Hd = np.zeros((A, H + 1), np.float32)
    V = np.ones((A, H + 1), bool)
    if mode == 'true':
        for a in range(A):
            j0 = t            # 严格从当前帧开始
            P[a] = sc.ap[a, j0:j0 + H + 1]
            Hd[a] = sc.ah[a, j0:j0 + H + 1]
            V[a] = sc.av[a, j0:j0 + H + 1]
        return P, Hd, V
    k = min(3, t)
    for a in range(A):
        p0 = sc.ap[a, t]
        v = (sc.ap[a, t] - sc.ap[a, t - k]) / (k * 0.1)
        sp = np.linalg.norm(v)
        hd = float(np.arctan2(v[1], v[0])) if sp > 0.3 else float(sc.ah[a, t])
        for j in range(H + 1):
            P[a, j] = p0 + v * j * 0.1
            Hd[a, j] = hd
        V[a, :] = sc.av[a, t]
    return P, Hd, V


def judge(sc, traj, P, Hd, V, offset):
    """offset=1：traj[0] 对应 t+1（规划轨迹）；offset=0：traj[0] 对应 t（推演）。"""
    H = min(len(traj), sc.n_future)
    traj = traj[:H]
    ec = min((_min_dist(p, sc.ea, sc.eb) for p in traj), default=1e9) \
        if len(sc.ea) else 1e9
    coll = False
    for tt in range(H):
        p = traj[tt]
        j = tt + offset
        if j >= P.shape[1]:
            break
        dv = traj[tt + 1] - p if tt + 1 < H else p - traj[tt - 1]
        h = float(np.arctan2(dv[1], dv[0])) if np.linalg.norm(dv) > 1e-6 else 0.0
        for a in range(len(P)):
            if not V[a, j]:
                continue
            q = P[a, j]
            if np.linalg.norm(q - p) > 6.0:
                continue
            if _obb_overlap(p, h, sc.ego_dim, q, float(Hd[a, j]), sc.ad[a]):
                coll = True
                break
        if coll:
            break
    return dict(ok=(ec >= ROAD_MARGIN and not coll), coll=coll,
                off=ec < ROAD_MARGIN, ec=float(ec))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--files-json', required=True)
    ap.add_argument('--n-scenes', type=int, default=2000)
    ap.add_argument('--k-top', type=int, default=5)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--out', default='/workspace/data/audit')
    args = ap.parse_args()

    import torch
    from train_head import Head

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
    print(f'场景 {len(names)} seed={args.seed}', flush=True)

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
        V_ = {m: agents_view(sc, m) for m in ('cv', 'true')}
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
            row = dict(scene=sid, rank=r, k=int(k), prob=float(prob[k]))
            for m in ('cv', 'true'):
                P, Hd, V = V_[m]
                jp = judge(sc, plan, P, Hd, V, offset=1)
                je = judge(sc, roll, P, Hd, V, offset=0)
                row[f'plan_{m}'] = jp
                row[f'roll_{m}'] = je
            # 软评分（PDM 式加权和），两种信息条件各算一次
            for m in ('cv', 'true'):
                je = row[f'roll_{m}']
                prog = float(sc.progress(roll[-1]))
                pen = (50.0 * int(je['coll']) + 5.0 * int(je['off'])
                       + 2.0 * max(0.0, 1.5 - min(3.0, je['ec'])) / 1.5)
                row[f'score_{m}'] = prog - pen
                row[f'prog'] = prog
            rows.append(row)
        if (i + 1) % 200 == 0:
            print(f'  {i+1}/{len(names)}', flush=True)

    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out, 'audit.npz'),
                        rows=json.dumps(rows))
    print('saved', args.out, flush=True)

    # ---------- 分析 ----------
    by = {}
    for r in rows:
        by.setdefault(r['scene'], []).append(r)
    for v in by.values():
        v.sort(key=lambda x: x['rank'])
    n = len(by)

    def pick_first(srows, key):
        for r in srows:
            if r[key]['ok']:
                return r
        return srows[0]

    def pick_score(srows, key):
        return max(srows, key=lambda r: r[key])

    strat = {'S1': 'first', 'S2': 'score'}
    res = {}
    for info in ('cv', 'true'):
        for sname in ('S1', 'S2'):
            coll = off = 0
            for sid, srows in by.items():
                if sname == 'S1':
                    r = pick_first(srows, f'roll_{info}')
                else:
                    r = pick_score(srows, f'score_{info}')
                je = r[f'roll_{info}']
                coll += int(je['coll'])
                off += int(je['off'])
            res[(sname, info)] = (coll, off)
    print('\n===== 失败数（碰撞 / 越界）=====')
    print(f"{'':8s} {'CV 预测':>16s} {'真实未来':>16s}")
    for sname in ('S1', 'S2'):
        c1, o1 = res[(sname, 'cv')]
        c2, o2 = res[(sname, 'true')]
        print(f'{sname:8s} {c1:7d} /{o1:5d}   {c2:7d} /{o2:5d}')
    g_cv = res[('S1', 'cv')][0] - res[('S2', 'cv')][0]
    g_tr = res[('S1', 'true')][0] - res[('S2', 'true')][0]
    print(f'\nS1 - S2 碰撞差：CV {g_cv:+d} | 真值 {g_tr:+d} | 变化 {g_tr-g_cv:+d}')


if __name__ == '__main__':
    main()
