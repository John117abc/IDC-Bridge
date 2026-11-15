"""Q1 补充：把横向位移按**车速**分层，看"高速时的横向避让"到底有多稀疏。

背景
----
Q1 测得 9.7% 的轨迹有 >1 m 的净横向位移。但 gap 场景归因显示，
那些"离车道 >2 m"的点 75% 处于静止状态 —— 所以这 10% 里可能很大一部分是
"从路侧/出入口并入"的低速场景，而不是高速行驶中的车道内避让。

本脚本把每条轨迹按速度切成"高速段"（v >= v_hi）和"低速段"（v <= v_lo），
在每段内部单独算净横向位移 |Δoffset|，从而回答：
    高速行驶时，真正发生 >1 m 横向机动的比例是多少？

用法（容器内，不需要 GPU）：
    python tools/check_lateral_by_speed.py --data-dir /data/data_json/training \
        --num-files 300 --seed 0 --v-hi 8 --v-lo 3
"""
import argparse
import glob
import json
import os

import numpy as np

DT = 0.1
LANE = "lane"


def polyline(arr):
    return np.asarray([[q["x"], q["y"]] for q in arr], dtype=np.float64)


def prep_lanes(roads):
    out = []
    for r in roads:
        if r.get("type") != LANE or len(r.get("geometry", [])) < 3:
            continue
        L = polyline(r["geometry"])
        if len(L) < 3:
            continue
        A, B = L[:-1], L[1:]
        ab = B - A
        den = np.maximum((ab ** 2).sum(1), 1e-12)
        out.append((L, A, B, ab, den,
                    float(np.linalg.norm(ab, axis=1).sum())))
    return out


def nearest_lane(q, lanes):
    best_d, best_proj, best_tan, best_arc, best_s = 1e9, None, None, None, None
    for (L, A, B, ab, den, arc) in lanes:
        t = np.clip(((q - A) * ab).sum(1) / den, 0.0, 1.0)
        proj = A + t[:, None] * ab
        d = np.linalg.norm(q - proj, axis=1)
        j = int(np.argmin(d))
        if d[j] < best_d:
            seg_len = np.linalg.norm(ab, axis=1)
            s = float(seg_len[:j].sum() + t[j] * seg_len[j])
            best_d = float(d[j])
            best_proj = proj[j]
            best_tan = ab[j] / max(float(seg_len[j]), 1e-9)
            best_arc = arc
            best_s = s
    return best_d, best_proj, best_tan, best_arc, best_s


def signed_lat(q, proj_pt, tan):
    n = np.array([-tan[1], tan[0]])
    return float((q - proj_pt) @ n)


def runs(mask, min_len):
    """连续 True 段（长度 >= min_len）的 (start, end) 列表。"""
    out, s = [], None
    for i, v in enumerate(mask):
        if v and s is None:
            s = i
        elif not v and s is not None:
            if i - s >= min_len:
                out.append((s, i))
            s = None
    if s is not None and len(mask) - s >= min_len:
        out.append((s, len(mask)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--num-files", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--v-hi", type=float, default=8.0)
    ap.add_argument("--v-lo", type=float, default=3.0)
    ap.add_argument("--min-len", type=int, default=12)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, "tfrecord-*.json")))
    rng = np.random.default_rng(args.seed)
    if len(files) > args.num_files:
        files = [files[i] for i in rng.choice(len(files), args.num_files,
                                              replace=False)]

    hi_d, lo_d, hi_v, lo_v = [], [], [], []
    traj_vmean, traj_d = [], []
    n_scene = 0
    for fp in files:
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        lanes = prep_lanes(d.get("roads") or [])
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

        # 逐点：速度 + 相对参考车道的横向偏移（含折线端点/航向截断）
        sp = np.r_[np.linalg.norm(np.diff(P, axis=0), axis=1) / DT, 0.0]
        off = np.full(len(P), np.nan)
        q0 = P[0]
        cand = []
        for li, ln in enumerate(lanes):
            d0 = nearest_lane(q0, [ln])[0]
            cand.append((d0, li))
        cand.sort()
        ref = lanes[cand[0][1]]
        arc = ref[5]
        hcos = np.cos(np.deg2rad(45.0))
        for i in range(len(P)):
            dd, proj_pt, tan, _arc, s = nearest_lane(P[i], [ref])
            if s <= 0.3 or s >= arc - 0.3:
                break
            if i + 1 < len(P):
                dseg = P[i + 1] - P[i]
                nn = np.linalg.norm(dseg)
                if nn > 1e-6 and abs(float(dseg @ tan) / nn) < hcos:
                    break
            off[i] = signed_lat(P[i], proj_pt, tan)
        good = np.isfinite(off)
        if good.sum() < 20:
            continue
        n_scene += 1
        traj_vmean.append(float(sp[good].mean()))
        traj_d.append(float(abs(off[good][-1] - off[good][0])))

        for mask, store_d, store_v in (
                (sp >= args.v_hi, hi_d, hi_v),
                (sp <= args.v_lo, lo_d, lo_v)):
            m = mask & good
            for (a, b) in runs(m, args.min_len):
                seg = off[a:b]
                seg = seg[np.isfinite(seg)]
                if len(seg) < args.min_len:
                    continue
                store_d.append(float(abs(seg[-1] - seg[0])))
                store_v.append(float(np.nanmean(sp[a:b])))

    if not traj_d:
        raise SystemExit("没有有效场景")

    def pct(x, ps=(50, 75, 90, 95, 99)):
        x = np.asarray(x)
        if not len(x):
            return "-"
        return "  ".join(f"p{p}={np.percentile(x, p):.2f}" for p in ps)

    hi_d, lo_d = np.asarray(hi_d), np.asarray(lo_d)
    hi_v, lo_v = np.asarray(hi_v), np.asarray(lo_v)
    td = np.asarray(traj_d)
    tv = np.asarray(traj_vmean)

    print(f"场景 {n_scene}")
    print(f"轨迹平均速度: {pct(tv, (10,50,90))} m/s")
    print()
    print(f"===== 高速段 (v >= {args.v_hi:.0f} m/s) =====")
    print(f"段数 {len(hi_d)}，段均速 {pct(hi_v, (50,90))}")
    print(f"段内净横向位移 |Δ|: {pct(hi_d)}")
    for th in (0.3, 0.5, 1.0, 1.5):
        print(f"   >{th:.1f} m: {100*np.mean(hi_d>th):5.1f}%")
    print()
    print(f"===== 低速段 (v <= {args.v_lo:.0f} m/s) =====")
    print(f"段数 {len(lo_d)}")
    print(f"段内净横向位移 |Δ|: {pct(lo_d)}")
    for th in (0.3, 0.5, 1.0, 1.5):
        print(f"   >{th:.1f} m: {100*np.mean(lo_d>th):5.1f}%")
    print()
    print("===== 对比 =====")
    if len(hi_d) and len(lo_d):
        print(f"  高速段 |Δ| 中位 {np.median(hi_d):.2f} m vs "
              f"低速段 {np.median(lo_d):.2f} m")
        print(f"  高速段 >1.0 m 占比 {100*np.mean(hi_d>1.0):.1f}% vs "
              f"低速段 {100*np.mean(lo_d>1.0):.1f}%")
    r = np.corrcoef(tv, td)[0, 1] if len(tv) > 3 else float("nan")
    print(f"  轨迹级：平均速度 与 净横向位移 的相关系数 = {r:.3f}")
    print()
    print("=== 判定 ===")
    if len(hi_d):
        q = 100 * np.mean(hi_d > 1.0)
        if q < 5:
            print(f"  高速时 >1 m 的横向机动只有 {q:.1f}% → **横向避让样本极稀疏**，")
            print("  必须靠'候选 + 代价'放大其作用，不能指望从数据里直接模仿")
        elif q < 15:
            print(f"  高速时 >1 m 的横向机动约 {q:.1f}% → 存在但不多")
        else:
            print(f"  高速时 >1 m 的横向机动 {q:.1f}% → 样本充足")

    if args.out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(args.out, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        bins = np.linspace(0, 3, 40)
        axes[0].hist(hi_d, bins=bins, alpha=0.7, label=f"v>={args.v_hi:.0f} (n={len(hi_d)})",
                     color="#3182bd")
        axes[0].hist(lo_d, bins=bins, alpha=0.7, label=f"v<={args.v_lo:.0f} (n={len(lo_d)})",
                     color="#e6550d")
        axes[0].axvline(1.0, color="#a50f15", ls="--", lw=1.2)
        axes[0].set_xlabel("|net lateral shift| within segment (m)")
        axes[0].set_ylabel("count"); axes[0].legend(fontsize=8)
        axes[0].set_title("lateral maneuver by speed regime"); axes[0].grid(alpha=0.25)
        axes[1].scatter(tv, td, s=8, alpha=0.6, c="#6a51a3")
        axes[1].axhline(1.0, color="#a50f15", ls="--", lw=1.0)
        axes[1].set_xlabel("mean speed of trajectory (m/s)")
        axes[1].set_ylabel("|net lateral shift| (m)")
        axes[1].set_title(f"per-trajectory (r={r:.2f})"); axes[1].grid(alpha=0.25)
        fig.tight_layout()
        fn = os.path.join(args.out, "lateral_by_speed.png")
        fig.savefig(fn, dpi=120); plt.close(fig)
        print("saved", fn)


if __name__ == "__main__":
    main()
