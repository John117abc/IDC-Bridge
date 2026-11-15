"""车道覆盖度诊断：单独一条 lane 折线能覆盖多长？一条轨迹要跨几条？路口在哪里？

回答两个问题：
  1. **有完整车道吗**：lane 折线弧长分布；一条 9.1 s 轨迹需要几条折线首尾相接
  2. **路口怎么办**：有多少采样点"离任何车道都 > 2 m"（可视为路口/待转区），
     这些点上车道参考失效，需要回退到 expert 路径

用法（容器内，不需要 GPU）：
    python tools/check_lane_coverage.py --data-dir /data/data_json/training \
        --num-files 200 --seed 0
"""
import argparse
import glob
import json
import os

import numpy as np

LANE = "lane"


def polyline(arr):
    return np.asarray([[q["x"], q["y"]] for q in arr], dtype=np.float64)


def dist_and_lane(q, lanes):
    """q 到所有 lane 折线的最近距离 + 最近 lane 的编号。"""
    best_d, best_i = 1e9, -1
    for i, (L, A, B, ab, den, _arc) in enumerate(lanes):
        t = np.clip(((q - A) * ab).sum(1) / den, 0.0, 1.0)
        proj = A + t[:, None] * ab
        d = float(np.linalg.norm(q - proj, axis=1).min())
        if d < best_d:
            best_d, best_i = d, i
    return best_d, best_i


def prep_lanes(roads):
    out = []
    for r in roads:
        if r.get("type") != LANE or len(r.get("geometry", [])) < 2:
            continue
        L = polyline(r["geometry"])
        if len(L) < 3:
            continue
        A, B = L[:-1], L[1:]
        ab = B - A
        den = np.maximum((ab ** 2).sum(1), 1e-12)
        arc = float(np.linalg.norm(ab, axis=1).sum())
        out.append((L, A, B, ab, den, arc))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--num-files", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gap-th", type=float, default=2.0,
                    help="判定'不在车道上'的距离阈值 (m)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, "tfrecord-*.json")))
    rng = np.random.default_rng(args.seed)
    if len(files) > args.num_files:
        files = [files[i] for i in rng.choice(len(files), args.num_files,
                                              replace=False)]

    arcs, n_spans, gap_frac, gap_len, traj_len = [], [], [], [], []
    dists = []
    n_scene = 0
    for fp in files:
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        lanes = prep_lanes(d.get("roads") or [])
        if not lanes:
            continue
        arcs.extend([x[5] for x in lanes])
        objs = d.get("objects") or []
        if not objs:
            continue
        idx = (d.get("metadata") or {}).get("sdc_track_index")
        if idx is None:
            idx = next((i for i, o in enumerate(objs) if o.get("mark_as_expert")), 0)
        o = objs[int(idx)]
        P = np.asarray([[q["x"], q["y"]] for q in (o.get("position") or [])],
                       dtype=np.float64)
        V = np.asarray(o.get("valid") or [True] * len(P), dtype=bool) \
            & np.isfinite(P).all(1)
        if V.sum() < 20:
            continue
        first_bad = int(np.argmin(V)) if not V.all() else len(V)
        P = P[:first_bad]
        if len(P) < 20:
            continue
        n_scene += 1
        traj_len.append(float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()))

        seq = []
        dd = np.empty(len(P))
        for i in range(len(P)):
            dmin, li = dist_and_lane(P[i], lanes)
            dd[i] = dmin
            seq.append(li if dmin < args.gap_th else -1)
        dists.append(dd)
        seq = np.asarray(seq)
        # 连续同车道段数（-1 视为"不在车道上"，单独一段）
        spans = 1 + int(np.sum(seq[1:] != seq[:-1]))
        n_spans.append(spans)
        gap_frac.append(float(np.mean(dd > args.gap_th)))
        # 最长连续"不在车道"段（步数）
        best = cur = 0
        for v in (dd > args.gap_th):
            cur = cur + 1 if v else 0
            best = max(best, cur)
        gap_len.append(best)

    if not n_scene:
        raise SystemExit("没有有效场景")

    arcs = np.asarray(arcs)
    tl = np.asarray(traj_len)
    ns = np.asarray(n_spans)
    gf = np.asarray(gap_frac) * 100
    gl = np.asarray(gap_len)
    dd = np.concatenate(dists)

    def pct(x, ps=(50, 75, 90, 95, 99)):
        return "  ".join(f"p{p}={np.percentile(x, p):.1f}" for p in ps)

    print(f"场景 {n_scene}，lane 折线 {len(arcs)} 条")
    print()
    print("【1】lane 折线弧长 (m):")
    print(f"   {pct(arcs)}")
    print(f"   < 30 m 的占比: {100*np.mean(arcs<30):.1f}%   "
          f"< 60 m: {100*np.mean(arcs<60):.1f}%")
    print()
    print("【2】轨迹长度 (m):", pct(tl, (50, 90, 95)))
    print(f"   轨迹长度 > 单条 lane 中位弧长({np.percentile(arcs,50):.0f} m) "
          f"的占比: {100*np.mean(tl > np.percentile(arcs,50)):.0f}%")
    print()
    print("【3】一条轨迹跨几条 lane 折线（按最近车道切换计数）:")
    print(f"   {pct(ns, (50, 75, 90, 95))}")
    print(f"   只跨 1 条: {100*np.mean(ns==1):.0f}%   "
          f"跨 ≥2 条: {100*np.mean(ns>=2):.0f}%   "
          f"跨 ≥4 条: {100*np.mean(ns>=4):.0f}%")
    print()
    print(f"【4】离最近车道 > {args.gap_th:.1f} m 的点（路口/待转区候选）:")
    print(f"   每个场景的比例: p50={np.percentile(gf,50):.1f}%  "
          f"p90={np.percentile(gf,90):.1f}%  max={gf.max():.1f}%")
    print(f"   场景中完全无此类点的占比: {100*np.mean(gf==0):.0f}%")
    print(f"   最长连续无车道段: p50={np.percentile(gl,50):.0f} 步  "
          f"p90={np.percentile(gl,90):.0f} 步  max={gl.max()} 步")
    print(f"   逐点最近车道距离分布: {pct(dd, (50, 90, 95, 99))}")
    print()
    print("=== 结论 ===")
    print(f"   • 单条 lane 折线中位 {np.percentile(arcs,50):.0f} m，"
          f"{100*np.mean(ns>=2):.0f}% 的轨迹需要跨 ≥2 条 → **车道不是一整条，需要拼接**")
    print(f"   • {100*np.mean(gf>0):.0f}% 的场景存在'离车道 >{args.gap_th:.0f}m'的点"
          f"（中位占 {np.percentile(gf,50):.1f}% 的步数）→ **路口处车道参考会失效**")


if __name__ == "__main__":
    main()
