"""上层规划器奖励 = 执行后轨迹的 PDMS（与评测同一套代码）。检查点 1 用。"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, '/gpudrive/baselines/IDC-Bridge/src')
sys.path.insert(0, '/gpudrive/baselines/IDC-Bridge/tools')

from metrics.openloop_pdms import Scene, score_trajectory  # noqa
from exp_pre import pure_pursuit  # noqa


def ego_to_world(pts, p0, h0):
    c, s = np.cos(h0), np.sin(h0)
    R = np.array([[c, -s], [s, c]], np.float64)
    return pts.astype(np.float64) @ R.T + p0[None]


def densify(anchor, n=40):
    t_src = np.linspace(0, 4.0, len(anchor))
    t_dst = np.linspace(0, 4.0, n)
    return np.stack([np.interp(t_dst, t_src, anchor[:, 0]),
                     np.interp(t_dst, t_src, anchor[:, 1])], 1)


def lateral_offset(path_ego, o):
    d = np.gradient(path_ego, axis=0)
    n = np.stack([-d[:, 1], d[:, 0]], 1)
    n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
    u = np.linspace(0, 1, len(path_ego))[:, None]
    S = 6 * u**5 - 15 * u**4 + 10 * u**3
    return path_ego + o * S * n


def reward_of(sc, path_ego, ego_w, sigma, eps, ramp, v0=10.0):
    cand_w = ego_to_world(path_ego + sigma * ramp * eps[None], ego_w[:2], ego_w[2])
    try:
        roll = pure_pursuit(cand_w, v0)
    except Exception:
        roll = cand_w
    roll = np.asarray(roll, np.float64)
    H = sc.n_future
    if len(roll) < H:
        roll = np.vstack([roll, np.repeat(roll[-1:], H - len(roll), 0)])
    roll = roll[:H]
    r = score_trajectory(sc, roll)
    return r, roll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/workspace/data/planner5k')
    ap.add_argument('--data-dir', default='/data/data_json/training')
    ap.add_argument('--scene', type=int, default=0)
    ap.add_argument('--n-scenes', type=int, default=1)
    ap.add_argument('--g', type=int, default=4)
    ap.add_argument('--sigma', type=float, default=0.8)
    ap.add_argument('--offsets', default='0,-1.6,1.6,-3.5,3.5')
    args = ap.parse_args()

    z = np.load(f'{args.data}/train.npz', allow_pickle=True)
    offs = [float(x) for x in args.offsets.split(',')]
    idxs = (list(range(args.n_scenes)) if args.n_scenes > 1
            else [args.scene])
    rows = []
    for i in idxs:
        name = str(z['scene_id'][i])
        fp = os.path.join(args.data_dir, name)
        if not os.path.exists(fp):
            print(f'[skip] {name} 不存在')
            continue
        d = json.load(open(fp))
        sc = Scene(d, horizon=40)
        ego_w = z['ego_world'][i]
        base = densify(z['gt'][i])
        rng = np.random.default_rng(1000 + i)
        ramp = (np.linspace(0, 1, len(base)) ** 2)[:, None]
        if args.n_scenes == 1:
            print(f'scene {name}')
            print(f'{"k":>2s} {"off":>5s} {"g":>2s} {"PDMS":>6s} {"coll":>5s} '
                  f'{"off":>5s} {"prog":>6s} {"ttc":>6s}')
        for k, o in enumerate(offs):
            cand = lateral_offset(base, o)
            seen = []
            for gg in range(args.g):
                eps = rng.normal(0, 1.0, (2,))
                r, _ = reward_of(sc, cand, ego_w, args.sigma, eps, ramp)
                seen.append(r['driving_score'])
                rows.append((i, k, o, gg, r['driving_score']))
                if args.n_scenes == 1:
                    print(f'{k:2d} {o:5.1f} {gg:2d} {r["driving_score"]:6.1f} '
                          f'{r.get("nc", 0):5.2f} {r.get("dac", 0):5.2f} '
                          f'{r.get("ep", 0):6.2f} {r.get("ttc", 0):6.2f}')
            if len(set(np.round(seen, 2))) == 1 and args.n_scenes == 1:
                print(f'   [WARN] offset={o} 的变体 PDMS 完全相同')

    if args.n_scenes > 1:
        v = np.array([r[4] for r in rows])
        groups = {}
        for r in rows:
            groups.setdefault((r[0], r[1]), []).append(r[4])
        diff = sum(1 for g in groups.values() if len(set(np.round(g, 2))) > 1)
        print(f'=== {len(idxs)} 场景 × {len(offs)} 候选 × {args.g} 变体 = '
              f'{len(rows)} 条 ===')
        print(f'组内有差异: {diff}/{len(groups)} ({100*diff/max(len(groups),1):.0f}%)')
        print(f'PDMS: min={v.min():.1f} p50={np.median(v):.1f} max={v.max():.1f} '
              f'唯一值={len(set(np.round(v,1)))}')
        print(f'PDMS=0 的比例: {100*np.mean(v <= 0):.0f}%')


if __name__ == '__main__':
    main()
