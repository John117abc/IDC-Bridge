"""归因那 12% "存在离车道 >2m 的点" 的场景：是路口？环岛？停车场/车道口？

对每个 gap 点（离任何 lane 折线 > gap_th）记录：
  • ego 车速
  • 到各类道路元素（crosswalk / driveway / road_edge / road_line）的最近距离
  • 最近 lane 的朝向与 ego 航向的对齐度（|cos| 小 = 交叉/环岛）
  • 出现在轨迹的时间位置（归一化 0~1）

用法（容器内，不需要 GPU）：
    python tools/check_gap_scenes.py --data-dir /data/data_json/training \
        --num-files 300 --seed 0 --gap-th 2.0
"""
import argparse
import glob
import json
import os
from collections import Counter

import numpy as np

GAP_TH = 2.0


def polyline(arr):
    return np.asarray([[q["x"], q["y"]] for q in arr], dtype=np.float64)


def prep(roads, want_types):
    out = {}
    for r in roads:
        t = r.get("type")
        if t not in want_types or len(r.get("geometry", [])) < 2:
            continue
        L = polyline(r["geometry"])
        if len(L) < 2:
            continue
        A, B = L[:-1], L[1:]
        ab = B - A
        den = np.maximum((ab ** 2).sum(1), 1e-12)
        out.setdefault(t, []).append((L, A, B, ab, den))
    return out


def nearest(q, items):
    """返回 (最近距离, 最近折线在最近段上的切向) —— items 为 prep 的列表。"""
    best_d, best_tan = 1e9, None
    for (L, A, B, ab, den) in items:
        t = np.clip(((q - A) * ab).sum(1) / den, 0.0, 1.0)
        proj = A + t[:, None] * ab
        d = np.linalg.norm(q - proj, axis=1)
        j = int(np.argmin(d))
        if d[j] < best_d:
            best_d = float(d[j])
            seg = ab[j] / max(float(np.linalg.norm(ab[j])), 1e-9)
            best_tan = seg
    return best_d, best_tan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--num-files", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gap-th", type=float, default=GAP_TH)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, "tfrecord-*.json")))
    rng = np.random.default_rng(args.seed)
    if len(files) > args.num_files:
        files = [files[i] for i in rng.choice(len(files), args.num_files,
                                              replace=False)]

    speeds, d_cw, d_dw, d_edge, d_lane_all = [], [], [], [], []
    aligns, tpos = [], []
    n_scene = n_gap_scene = 0
    scene_info = []

    for fp in files:
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        roads = d.get("roads") or []
        P_all = prep(roads, {"lane", "crosswalk", "driveway", "road_edge"})
        lanes = P_all.get("lane")
        if not lanes:
            continue
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
        has_gap = False
        for i in range(len(P)):
            dl, tan = nearest(P[i], lanes)
            d_lane_all.append(dl)
            if dl <= args.gap_th:
                continue
            has_gap = True
            sp = (float(np.linalg.norm(P[i + 1] - P[i]) / 0.1)
                  if i + 1 < len(P) else 0.0)
            speeds.append(sp)
            tpos.append(i / max(1, len(P) - 1))
            if tan is not None and i + 1 < len(P):
                dseg = P[i + 1] - P[i]
                nn = np.linalg.norm(dseg)
                if nn > 1e-6:
                    aligns.append(abs(float(dseg @ tan) / nn))
            for tname, store in (("crosswalk", d_cw), ("driveway", d_dw),
                                 ("road_edge", d_edge)):
                items = P_all.get(tname)
                store.append(nearest(P[i], items)[0] if items else np.inf)
        if has_gap:
            n_gap_scene += 1
            scene_info.append((os.path.basename(fp)[:26],
                               len(lanes), len(P)))

    if not speeds:
        print("没有 gap 点")
        return

    sp = np.asarray(speeds)
    al = np.asarray(aligns) if aligns else np.array([1.0])
    tp = np.asarray(tpos)
    cw = np.asarray(d_cw)
    dw = np.asarray(d_dw)
    ed = np.asarray(d_edge)
    dll = np.asarray(d_lane_all)

    def pct(x, ps=(50, 90, 95, 99)):
        x = x[np.isfinite(x)]
        if not len(x):
            return "-"
        return "  ".join(f"p{p}={np.percentile(x, p):.1f}" for p in ps)

    print(f"场景 {n_scene}，其中有 gap 的 {n_gap_scene} "
          f"({100*n_gap_scene/max(n_scene,1):.0f}%)，gap 点 {len(sp)} 个")
    print()
    print(f"【gap 点的 ego 车速 (m/s)】 {pct(sp)}")
    print(f"   车速 <1 m/s 占比: {100*np.mean(sp<1):.0f}%   "
          f"<3: {100*np.mean(sp<3):.0f}%   >8: {100*np.mean(sp>8):.0f}%")
    print()
    print(f"【到各元素距离 (m)】")
    print(f"   crosswalk: {pct(cw)}   <5m: {100*np.mean(cw<5):.0f}%  "
          f"<10m: {100*np.mean(cw<10):.0f}%")
    print(f"   driveway : {pct(dw)}   <5m: {100*np.mean(dw<5):.0f}%  "
          f"<10m: {100*np.mean(dw<10):.0f}%")
    print(f"   road_edge: {pct(ed)}")
    print()
    print(f"【最近 lane 与 ego 航向对齐度 |cos|】 {pct(al, (10,50,90))}")
    print(f"   |cos| < 0.5（垂直，像交叉/环岛）: {100*np.mean(al<0.5):.0f}%")
    print(f"   |cos| > 0.9（同向，只是横向偏了）: {100*np.mean(al>0.9):.0f}%")
    print()
    print(f"【gap 在轨迹中的位置（0=起点, 1=终点）】 {pct(tp, (10,50,90))}")
    print()
    print("=== 归因 ===")
    low_speed = np.mean(sp < 1)
    near_dw = np.mean(dw < 5)
    near_cw = np.mean(cw < 5)
    perp = np.mean(al < 0.5)
    if low_speed > 0.5 and near_dw > 0.4:
        print("   → 主要是**停车场/车道口**（低速 + 近 driveway）")
    elif near_cw > 0.4 or perp > 0.4:
        print("   → 主要是**交叉路口/环岛**（近人行横道 或 车道朝向垂直）")
    elif np.mean(sp > 8) > 0.5:
        print("   → 高速通过，更像**地图标注在主干道上断开**，不是停车场")
    else:
        print("   → 混合，无明显单一成因")
    print()
    print("有 gap 的场景示例（文件名 / lane 数 / 轨迹步数）:")
    for s in scene_info[:8]:
        print(f"   {s[0]:28s} lanes={s[1]:4d} steps={s[2]:3d}")


if __name__ == "__main__":
    main()
