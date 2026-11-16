"""按"8 秒未来窗口"测锚点集：oracle PDMS + 全零场景归因。

决策时刻 = frame 10（t=1.0 s）；预测/评测窗口 = frame 11..90（80 步）。
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

from metrics.openloop_pdms import Scene, score_trajectory  # noqa: E402

TYPE_MAP = {'road_edge': 1, 'road_line': 2, 'lane': 3}


def _poly(arr):
    return np.asarray([[q['x'], q['y']] for q in arr], dtype=np.float64)


def json_map_rows(d):
    rows = []
    for r in (d.get('roads') or []):
        t = TYPE_MAP.get(r.get('type'))
        if t is None or len(r.get('geometry') or []) < 2:
            continue
        L = _poly(r['geometry'])
        for a, b in zip(L[:-1], L[1:]):
            mid = 0.5 * (a + b)
            ln = float(np.linalg.norm(b - a))
            if ln < 1e-6:
                continue
            h = float(np.arctan2(b[1] - a[1], b[0] - a[0]))
            rows.append([mid[0], mid[1], ln, 0.0, 0.0, h, float(t)])
    return np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 7))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--n-scenes", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from env.candidate_sampler import SamplerConfig
    from env.map_candidates import generate_map_candidates

    scfg = SamplerConfig()
    files = sorted(glob.glob(os.path.join(args.data_dir, 'tfrecord-*.json')))
    rng = np.random.default_rng(args.seed)
    if len(files) > args.n_scenes:
        files = [files[i] for i in rng.choice(len(files), args.n_scenes,
                                              replace=False)]

    K = len(scfg.lat_offsets) * len(scfg.anchor_lengths)
    print(f"锚点 {len(scfg.lat_offsets)} 横向 × {len(scfg.anchor_lengths)} 长度 = {K} 条")
    print(f"窗口: 历史 0..9, 现在 10, 未来 11..90 (80 步 ≈ 8 s)")

    oracle, center_best, mean_all = [], [], []
    n_ok = n_col = n_off = n_both = 0
    win_off, win_len = [], []
    for fp in files:
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        sc = Scene(d)
        if sc.n_future < 20:
            continue
        rows = json_map_rows(d)
        if len(rows) == 0:
            continue
        pose = sc.pose_now()
        v0 = max(sc.speed_now(), 1.0)
        cands, _b = generate_map_candidates(pose, rows, scfg.lat_offsets, scfg,
                                            speed0=v0, n=sc.n_future)
        res = []
        for c in cands:
            r = score_trajectory(sc, c.pos)
            if r.get('valid'):
                res.append((c, r))
        if not res:
            continue
        scores = np.array([r['driving_score'] for _, r in res])
        oracle.append(float(scores.max()))
        mean_all.append(float(scores.mean()))
        j = int(np.argmax(scores))
        win_off.append(res[j][0].param['lat_offset'])
        win_len.append(res[j][0].param['len_target'])
        c_idx = [i for i, (c, _) in enumerate(res)
                 if abs(c.param['lat_offset']) < 1e-9]
        if c_idx:
            center_best.append(float(max(scores[i] for i in c_idx)))
        max_nc = max(r['nc'] for _, r in res)
        max_dac = max(r['dac'] for _, r in res)
        if max_nc > 0.5 and max_dac > 0.5:
            n_ok += 1
        elif max_nc < 0.5 and max_dac < 0.5:
            n_both += 1
        elif max_nc < 0.5:
            n_col += 1
        else:
            n_off += 1

    if not oracle:
        print("无样本")
        return
    O = np.asarray(oracle)
    tot = n_ok + n_col + n_off + n_both
    print()
    print(f"样本 {tot}")
    print(f"  oracle PDMS: mean={O.mean():.1f} p10={np.percentile(O,10):.1f} "
          f"p50={np.percentile(O,50):.1f} p90={np.percentile(O,90):.1f} "
          f"max={O.max():.1f}")
    print(f"  平均(随机锚点) = {np.mean(mean_all):.1f}")
    if center_best:
        print(f"  仅中心锚点     = {np.mean(center_best):.1f}  "
              f"(横向增益 {O.mean()-np.mean(center_best):+.1f})")
    print()
    print(f"  可用(有解)        : {n_ok:3d} ({100*n_ok/tot:4.1f}%)")
    print(f"  全部撞车(NC 全 0) : {n_col:3d} ({100*n_col/tot:4.1f}%)")
    print(f"  全部压线(DAC 全 0): {n_off:3d} ({100*n_off/tot:4.1f}%)")
    print(f"  两者都全 0        : {n_both:3d} ({100*n_both/tot:4.1f}%)")
    print()
    print("获胜锚点 横向分布:", {o: int(np.sum(np.abs(np.asarray(win_off)-o) < 1e-6))
                              for o in sorted(set(np.round(win_off, 2)))})
    print("获胜锚点 长度分布:", {L: int(np.sum(np.abs(np.asarray(win_len)-L) < 1e-6))
                              for L in sorted(set(np.round(win_len, 1)))})


if __name__ == "__main__":
    main()
