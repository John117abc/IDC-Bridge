"""检查"全部撞车"场景里的碰撞是不是检测器假阳性。

对每个全撞场景，取一条候选，找到第一次判撞的时刻，打印：
  · 撞的是哪类 agent、它的尺寸
  · 自车系下的纵向/横向间距，与"两车半长/半宽之和"比较
  · 人类轨迹在同一场景里是否也被判撞
若大量事件的 |横向间距| 只比半宽和小一点点 → 检测器过松（假阳性）
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _ROOT)

from metrics.openloop_pdms import (EGO_DIM, SceneCache, _obb_overlap,  # noqa: E402
                                   score_trajectory, step_metrics)
from tools.oracle_anchor_pdms import json_map_rows  # noqa: E402


def first_collision(sc, traj):
    T = min(len(traj), sc.T)
    d = np.diff(traj[:T], axis=0)
    heading = np.r_[np.arctan2(d[:, 1], d[:, 0]), 0.0]
    heading[-1] = heading[-2]
    for t in range(T):
        p = traj[t]; h = float(heading[t])
        if not len(sc.ap):
            continue
        dd = np.linalg.norm(sc.ap[:, t, :] - p[None, :], axis=1)
        for a in np.where((dd < 6.0) & sc.av[:, t])[0]:
            if _obb_overlap(p, h, EGO_DIM, sc.ap[a, t], sc.ah[a, t], sc.ad[a]):
                dv = sc.ap[a, t] - p
                c, s = math.cos(-h), math.sin(-h)
                along = float(dv[0] * c - dv[1] * s)
                lat = float(dv[0] * s + dv[1] * c)
                return t, int(a), along, lat
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--n-scenes", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from env.candidate_sampler import SamplerConfig
    from env.map_candidates import generate_map_candidates

    scfg = SamplerConfig()
    cache = SceneCache(args.data_dir, n_scenes=args.n_scenes, seed=args.seed)
    files = sorted(glob.glob(os.path.join(args.data_dir, 'tfrecord-*.json')))
    rng = np.random.default_rng(args.seed)
    if len(files) > args.n_scenes:
        files = [files[i] for i in rng.choice(len(files), args.n_scenes,
                                              replace=False)]
    raw = []
    for fp in files:
        try:
            with open(fp) as f:
                raw.append(json.load(f))
        except Exception:
            raw.append(None)

    n_blocked = 0
    marginals = 0
    details = []
    for w, (sc, d) in enumerate(zip(cache.scenes, raw)):
        if d is None or len(sc.expert_pos) < 20:
            continue
        mrows = json_map_rows(d)
        if len(mrows) == 0:
            continue
        ex = sc.expert_pos
        if not np.all(np.abs(ex) < 1e4):
            continue
        d0 = ex[1] - ex[0]
        h0 = float(np.arctan2(d0[1], d0[0])) if np.linalg.norm(d0) > 0.05 \
            else float(sc.expert_heading[0])
        pose = (float(ex[0, 0]), float(ex[0, 1]), h0)
        v0 = float(np.linalg.norm(d0) / 0.1)
        cands, _b = generate_map_candidates(pose, mrows, scfg.lat_offsets, scfg,
                                            speed0=v0)
        res = [(c, score_trajectory(sc, c.pos)) for c in cands]
        res = [(c, r) for c, r in res if r.get('valid')]
        if not res:
            continue
        if max(r['nc'] for _, r in res) > 0.5:
            continue                       # 不是"全撞"场景
        n_blocked += 1
        # 人类轨迹是否也撞
        hm = sc.human
        human_coll = int(np.sum(hm['coll']))
        # 取"碰撞最早"的候选来看细节
        best = None
        for c, r in res:
            fc = first_collision(sc, c.pos)
            if fc is not None and (best is None or fc[0] < best[0]):
                best = (fc[0], fc[1], fc[2], fc[3], c)
        if best is None:
            continue
        t, a, along, lat, c = best
        half_L = EGO_DIM[0] / 2 + sc.ad[a][0] / 2
        half_W = EGO_DIM[1] / 2 + sc.ad[a][1] / 2
        margin = max(abs(lat) - half_W, abs(along) - half_L)
        marginal = margin > -0.3
        marginals += int(marginal)
        details.append((w, t, a, along, lat, half_L, half_W, margin,
                        human_coll, c.param['lat_offset'],
                        c.param['len_target']))

    print(f"\n全部撞车的场景: {n_blocked}")
    print(f"{'w':>3s} {'t':>4s} {'agent':>5s} {'along':>7s} {'lat':>7s} "
          f"{'halfL':>6s} {'halfW':>6s} {'margin':>7s} {'人撞步':>6s} "
          f"{'o':>5s} {'L':>6s}  判定")
    for (w, t, a, along, lat, hL, hW, mg, hc, o, L) in details:
        verdict = "假阳性(临界)" if mg > -0.3 else "真碰撞"
        print(f"{w:3d} {t:4d} {a:5d} {along:7.2f} {lat:7.2f} {hL:6.2f} {hW:6.2f} "
              f"{mg:7.2f} {hc:6d} {o:5.1f} {L:6.1f}  {verdict}")
    if n_blocked:
        print(f"\n临界(可能假阳性)占比: {marginals}/{n_blocked} "
              f"= {100*marginals/max(n_blocked,1):.0f}%")


if __name__ == "__main__":
    main()
