"""Q1-v4：直接读原始 JSON 的 lane 折线，做真正的车道关联来测横向行为。

为什么必须回到原始 JSON
----------------------
GPUDrive 的 `map_observation_tensor` 把每条车道折线打散成了**互不关联的线段**
（type=lane_center, id 只覆盖约 3 段），所以"最近的 lane 分段"会频繁跳变，
导致 v1/v2/v3 全部失真：
    v1 净横向位移 p50=3.65 m / 车道 id 切换 100%
    v2 走廊宽 13.5 m / 横移速度 p99=26 m/s
    v3 走廊宽 21.1 m (p90=104.8)

原始 JSON 保留了结构：
    roads[i] = {geometry: [{x,y,z}...]（连续折线）, type: 'lane'|'road_edge'|..., id}
    objects[i] = {position:[91], heading:[91], velocity:[91], valid:[91], mark_as_expert}
    metadata.sdc_track_index -> ego 的下标

本脚本不需要 GPU：直接解析 JSON，用 lane 折线做参考，测 ego 的横向偏移。

用法：
    python tools/check_lateral_raw.py --data-dir /data/data_json/training \
        --num-files 300 --out /workspace/data/plots/lateral_raw
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

LANE_TYPE = "lane"
MAX_LAT = 6.0          # 参考车道搜索半径（m）


def polyline(arr):
    p = np.asarray([[q["x"], q["y"]] for q in arr], dtype=np.float64)
    return p


def point_to_polyline_dist(q, P):
    """q [2] 到折线 P [M,2] 的最近距离、最近点、沿线弧长、局部切向。"""
    A, B = P[:-1], P[1:]
    ab = B - A
    den = np.maximum((ab ** 2).sum(1), 1e-12)
    t = np.clip(((q - A) * ab).sum(1) / den, 0.0, 1.0)
    proj = A + t[:, None] * ab
    d = np.linalg.norm(q - proj, axis=1)
    j = int(np.argmin(d))
    seg_len = np.linalg.norm(ab, axis=1)
    s = float(seg_len[:j].sum() + t[j] * seg_len[j])
    tan = ab[j] / max(seg_len[j], 1e-9)
    return float(d[j]), proj[j], s, tan


def signed_lat(q, proj_pt, tan):
    n = np.array([-tan[1], tan[0]])         # 折线左法向
    return float((q - proj_pt) @ n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--num-files", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot-n", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, "tfrecord-*.json")))
    if not files:
        raise SystemExit("没找到 json")
    rng = np.random.default_rng(args.seed)
    if len(files) > args.num_files:
        files = [files[i] for i in rng.choice(len(files), args.num_files,
                                              replace=False)]

    all_off, per, ex = [], [], []
    n_lane_poly, lane_arc = [], []
    n_scene = 0
    for fp in files:
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        roads = d.get("roads") or []
        lanes = [polyline(r["geometry"]) for r in roads
                 if r.get("type") == LANE_TYPE and len(r.get("geometry", [])) >= 2]
        lanes = [L for L in lanes if len(L) >= 3]
        if not lanes:
            continue
        n_lane_poly.append(len(lanes))
        objs = d.get("objects") or []
        meta = d.get("metadata") or {}
        idx = meta.get("sdc_track_index")
        if idx is None:
            idx = next((i for i, o in enumerate(objs) if o.get("mark_as_expert")),
                       0)
        if not objs:
            continue
        obj = objs[int(idx)]
        pos = obj.get("position") or []
        valid = obj.get("valid") or [True] * len(pos)
        P = np.asarray([[q["x"], q["y"]] for q in pos], dtype=np.float64)
        V = np.asarray(valid, dtype=bool) & np.isfinite(P).all(1)
        if V.sum() < 20:
            continue
        first_bad = int(np.argmin(V)) if not V.all() else len(V)
        P = P[:first_bad]
        if len(P) < 20:
            continue

        # 选参考车道：在起点附近挑最近的 lane 折线
        q0 = P[0]
        best, bestd = None, 1e9
        for L in lanes:
            d0, proj0, s0, tan0 = point_to_polyline_dist(q0, L)
            if d0 < bestd:
                best, bestd = L, d0
        if best is None or bestd > MAX_LAT:
            continue
        L = best
        arc = float(np.linalg.norm(np.diff(L, axis=0), axis=1).sum())
        lane_arc.append(arc)

        offs = np.full(len(P), np.nan)
        hcos = np.cos(np.deg2rad(45.0))
        for i in range(len(P)):
            dd, proj_pt, s, tan = point_to_polyline_dist(P[i], L)
            # 截断：跑出参考折线的弧长范围，或航向偏离 > 45°，就不再测量
            # （否则最近点被卡在折线端点上，offset 会虚增到几十米）
            if s <= 0.3 or s >= arc - 0.3:
                break
            if i + 1 < len(P):
                dseg = P[min(i + 1, len(P) - 1)] - P[i]
                nn = np.linalg.norm(dseg)
                if nn > 1e-6 and abs(float(dseg @ tan) / nn) < hcos:
                    break
            offs[i] = signed_lat(P[i], proj_pt, tan)
        v = np.isfinite(offs)
        if v.sum() < 20:
            continue
        o = offs[v]
        n_scene += 1
        all_off.append(o)
        per.append(dict(delta=float(o[-1] - o[0]), max_abs=float(np.abs(o).max()),
                        swing=float(o.max() - o.min())))
        if len(ex) < args.plot_n and abs(o[-1] - o[0]) > 1.0:
            ex.append((os.path.basename(fp)[:28], np.arange(v.sum()) * 0.1, o))

    if not per:
        raise SystemExit("没有有效场景")

    off = np.concatenate(all_off)
    A = np.abs(off)
    dl = np.abs(np.asarray([r["delta"] for r in per]))
    mx = np.asarray([r["max_abs"] for r in per])
    sw = np.asarray([r["swing"] for r in per])
    dots = np.concatenate([np.abs(np.diff(o)) / 0.1 for o in all_off
                           if len(o) > 5])

    def pct(x, ps=(50, 75, 90, 95, 99)):
        return "  ".join(f"p{p}={np.percentile(x, p):.2f}" for p in ps)

    print(f"场景 {n_scene}（来自 {len(files)} 个文件），采样点 {len(off)}")
    print(f"每条车道的折线数: p50={np.percentile(n_lane_poly,50):.0f}  "
          f"折线弧长 p50={np.percentile(lane_arc,50):.1f} m  "
          f"p90={np.percentile(lane_arc,90):.1f} m")
    print()
    print("【1】相对参考车道中心线的横向偏移 |offset| (m):")
    print(f"   {pct(A)}")
    print(f"   >0.5: {100*np.mean(A>0.5):.1f}%  >1.0: {100*np.mean(A>1.0):.1f}%  "
          f">1.5: {100*np.mean(A>1.5):.1f}%")
    print()
    print("【2】单条轨迹 (m):")
    print(f"   最大|offset|:  {pct(mx)}")
    print(f"   净横向位移|Δ|: {pct(dl)}")
    print(f"   横向摆动幅度:  {pct(sw)}")
    print()
    print("【3】横向机动事件:")
    for th in (0.2, 0.3, 0.5, 1.0, 1.5, 3.0):
        print(f"   |Δoffset| > {th:.1f} m: {100*np.mean(dl > th):5.1f}%")
    print()
    print("【4】横移速度 (m/s):", pct(dots))
    print()
    print("=== 可信性自检 ===")
    ok_v = np.percentile(dots, 99) < 3.0
    print(f"   横移速度 p99 = {np.percentile(dots,99):.2f} m/s -> "
          f"{'合理' if ok_v else '异常'}")
    if ok_v:
        q = np.percentile(dl, 95)
        if q < 0.3:
            print("   => 几乎没有横向机动")
        elif q < 1.0:
            print(f"   => 有小幅横向机动（p95={q:.2f} m）")
        else:
            print(f"   => 有明显横向机动（p95={q:.2f} m）")

    if args.out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(args.out, exist_ok=True)
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
        axes[0, 0].hist(off, bins=100, color="#3182bd")
        axes[0, 0].set_xlabel("lateral offset to lane centerline (m)")
        axes[0, 0].set_ylabel("count")
        axes[0, 0].set_title(f"all points (lane polyline reference, n={n_scene})")
        axes[0, 0].grid(alpha=0.25)
        axes[0, 1].hist(dl, bins=60, color="#e6550d")
        for th, c in ((0.5, "#a50f15"), (1.0, "#54278f"), (3.0, "#006d2c")):
            axes[0, 1].axvline(th, color=c, ls="--", label=f"{th} m")
        axes[0, 1].set_xlabel("|net lateral shift| per trajectory (m)")
        axes[0, 1].set_title("lateral maneuver magnitude")
        axes[0, 1].legend(fontsize=8); axes[0, 1].grid(alpha=0.25)
        xs = np.sort(dl)
        axes[1, 0].plot(xs, np.linspace(0, 1, len(xs)), lw=2, color="#31a354")
        for th in (0.5, 1.0, 3.0):
            axes[1, 0].axvline(th, color="#999999", ls=":", lw=0.8)
        axes[1, 0].set_xlabel("|net lateral shift| (m)")
        axes[1, 0].set_ylabel("CDF"); axes[1, 0].set_title("CDF")
        axes[1, 0].grid(alpha=0.25)
        for (nm, tt, o) in ex[:4]:
            axes[1, 1].plot(tt, o, lw=1.5, label=nm)
        axes[1, 1].axhline(0, color="#999999", lw=0.8)
        axes[1, 1].set_xlabel("time (s)")
        axes[1, 1].set_ylabel("offset (m)")
        axes[1, 1].set_title("examples with |Δ|>1.0 m")
        axes[1, 1].legend(fontsize=6); axes[1, 1].grid(alpha=0.25)
        fig.tight_layout()
        fn = os.path.join(args.out, "lateral_offset.png")
        fig.savefig(fn, dpi=120); plt.close(fig)
        print("saved", fn)


if __name__ == "__main__":
    main()
