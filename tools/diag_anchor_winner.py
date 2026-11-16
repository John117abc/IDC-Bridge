"""诊断：为什么 o=-3.5 的锚点常常赢？

对每个场景比较"最优锚点"与"最优中心锚点(o=0)"的分项差值：
    Δ = score(winner) - score(center)
若 ΔLK 主导 → 指标漏洞（LK 只认"最近车道"，跳到隔壁车道也能满分）
若 ΔEP/ΔNC 主导 → 路由问题（分叉走错）
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _ROOT)

from metrics.openloop_pdms import SceneCache, score_trajectory  # noqa: E402
from tools.oracle_anchor_pdms import json_map_rows  # noqa: E402

COMPS = ('nc', 'dac', 'ep', 'ttc', 'comfort', 'lk')


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

    rows = []
    for sc, d in zip(cache.scenes, raw):
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
        res = []
        for c in cands:
            r = score_trajectory(sc, c.pos)
            if r.get('valid'):
                res.append((c, r))
        if not res:
            continue
        i_win = int(np.argmax([r['driving_score'] for _, r in res]))
        c_idx = [i for i, (c, _) in enumerate(res)
                 if abs(c.param['lat_offset']) < 1e-9]
        if not c_idx:
            continue
        i_c = max(c_idx, key=lambda i: res[i][1]['driving_score'])
        rows.append((res[i_win][0].param['lat_offset'],
                     res[i_win][1], res[i_c][1]))

    if not rows:
        print("无样本")
        return
    print(f"样本 {len(rows)}")
    print()
    print(f"{'获胜 o':>8s} {'score_win':>10s} {'score_center':>13s} "
          + " ".join(f"{c:>9s}" for c in COMPS))
    print(f"{'':>8s} {'':>10s} {'':>13s} "
          + " ".join(f"{'Δ':>9s}" for c in COMPS))
    dsum = {c: [] for c in COMPS}
    for o, w, cc in rows:
        deltas = {c: (w[c] - cc[c]) if c in ('ep', 'ttc', 'comfort', 'lk')
                  else (w[c] - cc[c]) for c in COMPS}
        for c in COMPS:
            dsum[c].append(deltas[c])
        print(f"{o:8.1f} {w['driving_score']:10.1f} "
              f"{cc['driving_score']:13.1f} "
              + " ".join(f"{deltas[c]:9.3f}" for c in COMPS))
    print()
    print("【平均分项差值（winner - center）】")
    for c in COMPS:
        m = float(np.mean(dsum[c]))
        if abs(m) > 1e-6:
            print(f"  Δ{c:8s} = {m:+.4f}")
    print()
    print("【哪个分项最能解释增益】")
    for c in COMPS:
        pos = int(np.sum(np.asarray(dsum[c]) > 0.05))
        neg = int(np.sum(np.asarray(dsum[c]) < -0.05))
        print(f"  {c:8s}: 改善 {pos:2d} 场 / 变差 {neg:2d} 场")


if __name__ == "__main__":
    main()
