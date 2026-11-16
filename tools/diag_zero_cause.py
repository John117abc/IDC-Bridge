"""归因：锚点集拿 0 分的场景，是"撞车"还是"压线"造成的？

对每个场景，把所有锚点都评一遍，取：
    max_nc  = max over candidates of NC   （有没有一条不撞）
    max_dac = max over candidates of DAC  （有没有一条不压线）
据此把场景分成四类，定位主要瓶颈。
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

    n_ok = n_col = n_off = n_both = 0
    best_scores, col_list, off_list = [], [], []
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
        res = [score_trajectory(sc, c.pos) for c in cands]
        res = [r for r in res if r.get('valid')]
        if not res:
            continue
        best = max(r['driving_score'] for r in res)
        best_scores.append(best)
        max_nc = max(r['nc'] for r in res)
        max_dac = max(r['dac'] for r in res)
        max_ddc = max(r['ddc'] for r in res)
        blocked_nc = max_nc < 0.5
        blocked_dac = max_dac < 0.5
        blocked_ddc = max_ddc < 0.5
        if not (blocked_nc or blocked_dac or blocked_ddc):
            n_ok += 1
        elif blocked_nc and blocked_dac:
            n_both += 1
            col_list.append(sc)
        elif blocked_nc:
            n_col += 1
            col_list.append(sc)
        elif blocked_dac:
            n_off += 1
            off_list.append(sc)

    tot = n_ok + n_col + n_off + n_both
    print(f"\n样本 {tot}")
    print(f"  有一条可用锚点        : {n_ok:3d}  ({100*n_ok/max(tot,1):5.1f}%)")
    print(f"  全部撞车（NC 全 0）   : {n_col:3d}  ({100*n_col/max(tot,1):5.1f}%)")
    print(f"  全部压线（DAC 全 0）  : {n_off:3d}  ({100*n_off/max(tot,1):5.1f}%)")
    print(f"  两者都全 0            : {n_both:3d}  ({100*n_both/max(tot,1):5.1f}%)")
    if best_scores:
        S = np.asarray(best_scores)
        print(f"\nbest PDMS: mean={S.mean():.1f} p50={np.median(S):.1f} "
              f"max={S.max():.1f}")
    print()


if __name__ == "__main__":
    main()
