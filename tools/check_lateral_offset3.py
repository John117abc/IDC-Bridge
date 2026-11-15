"""Q1-v3：固定参考的横向偏移测量（连续性跟踪左右边界）。

前两版都栽在"参考跳变"上：
  v1（最近 lane_center 分段）：分段 id 只覆盖约 3 段/10 m，构不成连续车道，
      最近分段频繁跳 → 净横向位移 p50=3.65 m、车道 id 切换 100%
  v2（最近左右边界取中点）：最近边界会跳到另一条路/匝道/交叉口的边界，
      走廊宽度 p50=13.5 m、横移速度 p99=26 m/s（物理不可能）

v3 的做法：**把参考固定住**
  t=0 时按 ego 航向选左右各一条边界线段；
  之后每步只在"上一帧那条线段的邻域内"找同侧线段（按线段中点距离最近），
  找不到就沿用上一帧那条（对固定线段的点-线段距离始终有意义）。
  offset = (d_right - d_left)/2，参考系因此是连续且稳定的。

判读：横移速度 p99 若落到 2 m/s 以内、走廊宽度回到 7~10 m，说明度量可信。

用法（容器内）：
    python tools/check_lateral_offset3.py --data-dir /data/data_json/training \
        --num-worlds 40 --seed 0 --out /workspace/data/plots/lateral3
"""
import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import numpy as np  # noqa: E402

IX_X, IX_Y, IX_LEN, IX_HEAD, IX_TYPE = 0, 1, 2, 5, 6
ROAD_EDGE = 1
SEARCH_R = 12.0        # 连续性搜索半径（m）


def boundary_segments(rows):
    m = rows[rows[:, IX_TYPE].astype(int) == ROAD_EDGE]
    if not len(m):
        return None
    key = np.round(m[:, [IX_X, IX_Y, IX_LEN, IX_HEAD]], 4)
    _, idx = np.unique(key, axis=0, return_index=True)
    m = m[np.sort(idx)]
    x, y = m[:, IX_X], m[:, IX_Y]
    L, h = np.maximum(m[:, IX_LEN], 1e-3), m[:, IX_HEAD]
    c, s = np.cos(h), np.sin(h)
    a = np.stack([x - 0.5 * L * c, y - 0.5 * L * s], 1)
    b = np.stack([x + 0.5 * L * c, y + 0.5 * L * s], 1)
    mid = 0.5 * (a + b)
    return a, b, mid


def seg_dists(p, a, b):
    ab = b - a
    den = np.maximum((ab ** 2).sum(1), 1e-12)
    t = np.clip(((p - a) * ab).sum(1) / den, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return np.linalg.norm(p - proj, axis=1), proj


def side_of(p, h, proj):
    n = np.array([-np.sin(h), np.cos(h)])       # ego 左法向
    return (proj - p) @ n                        # >0 左侧


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--num-worlds", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot-n", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from gpudrive.env.dataset import SceneDataLoader
    from gpudrive.env.env_torch import GPUDriveTorchEnv

    from env.env_utils import get_ego_indices, get_env_config

    loader = SceneDataLoader(root=args.data_dir, batch_size=args.num_worlds,
                             dataset_size=args.num_worlds,
                             sample_with_replacement=False, shuffle=True,
                             seed=args.seed)
    env = GPUDriveTorchEnv(config=get_env_config(), data_loader=loader,
                           max_cont_agents=1, device="cuda",
                           action_type="continuous")
    env.reset()
    ego = get_ego_indices(env, args.num_worlds)
    maps = env.sim.map_observation_tensor().to_torch().cpu().numpy()
    raw = env.sim.expert_trajectory_tensor().to_torch().cpu().numpy()
    T = 91
    pos = raw[:, :, :2 * T].reshape(raw.shape[0], raw.shape[1], T, 2)

    all_off, per, ex = [], [], []
    n_scene = 0
    for w in range(args.num_worlds):
        rows = maps[w][np.abs(maps[w]).sum(1) > 1e-6]
        seg = boundary_segments(rows)
        if seg is None:
            continue
        a, b, mid = seg
        if len(a) < 5:
            continue
        p = pos[w, ego[w]]
        ok = (np.abs(p[:, 0]) < 1e3) & (np.abs(p[:, 1]) < 1e3)
        if not ok.any():
            continue
        first_bad = int(np.argmin(ok)) if not ok.all() else len(ok)
        p = p[:first_bad]
        if len(p) < 20:
            continue
        d = np.diff(p, axis=0)
        hs = np.r_[np.unwrap(np.arctan2(d[:, 1], d[:, 0])), 0.0]

        iL = iR = -1
        off = np.full(len(p), np.nan)
        wid = np.full(len(p), np.nan)
        for i in range(len(p)):
            dist, proj = seg_dists(p[i], a, b)
            sgn = side_of(p[i], hs[i], proj)
            if i == 0 or iL < 0 or iR < 0:
                candL = np.where((sgn > 0.2) & (dist < 30.0))[0]
                candR = np.where((sgn < -0.2) & (dist < 30.0))[0]
                if len(candL) == 0 or len(candR) == 0:
                    continue
                iL = int(candL[np.argmin(dist[candL])])
                iR = int(candR[np.argmin(dist[candR])])
            else:
                # 连续性：在同侧且中点距离上一帧中点 < SEARCH_R 的里取最近的
                for side, prev in ((+1, iL), (-1, iR)):
                    cand = np.where((sgn * side > 0.2)
                                    & (np.linalg.norm(mid - mid[prev], axis=1)
                                       < SEARCH_R))[0]
                    if len(cand):
                        k = int(cand[np.argmin(dist[cand])])
                        if side > 0:
                            iL = k
                        else:
                            iR = k
            dl = float(dist[iL])
            dr = float(dist[iR])
            off[i] = 0.5 * (dr - dl)
            wid[i] = dl + dr
        v = np.isfinite(off)
        if v.sum() < 20:
            continue
        n_scene += 1
        o = off[v]
        all_off.append(o)
        per.append(dict(w=w, max_abs=float(np.abs(o).max()),
                        delta=float(o[-1] - o[0]),
                        swing=float(o.max() - o.min()),
                        width=float(np.nanmedian(wid[v]))))
        if len(ex) < args.plot_n and abs(o[-1] - o[0]) > 1.0:
            ex.append((w, np.arange(v.sum()) * 0.1, o))

    if not per:
        raise SystemExit("没有有效场景")

    off = np.concatenate(all_off)
    A = np.abs(off)
    mx = np.asarray([r["max_abs"] for r in per])
    dl = np.abs(np.asarray([r["delta"] for r in per]))
    sw = np.asarray([r["swing"] for r in per])
    wd = np.asarray([r["width"] for r in per])
    dots = np.concatenate([np.abs(np.diff(o)) / 0.1 for o in all_off
                           if len(o) > 5])

    def pct(x, ps=(50, 75, 90, 95, 99)):
        return "  ".join(f"p{p}={np.percentile(x, p):.2f}" for p in ps)

    print(f"有效场景 {n_scene}，采样点 {len(off)}")
    print(f"走廊宽度（左右边界之和）: p50={np.percentile(wd,50):.1f} "
          f"p10={np.percentile(wd,10):.1f} p90={np.percentile(wd,90):.1f} m")
    print()
    print("【1】相对走廊中点的横向偏移 |offset| (m):")
    print(f"   {pct(A)}")
    print()
    print("【2】单条轨迹 (m):")
    print(f"   最大|offset|: {pct(mx)}")
    print(f"   净横向位移|Δ|: {pct(dl)}")
    print(f"   横向摆动幅度:  {pct(sw)}")
    print()
    print("【3】横向机动事件:")
    for th in (0.2, 0.3, 0.5, 1.0, 1.5):
        print(f"   |Δoffset| > {th:.1f} m: {100*np.mean(dl > th):5.1f}%")
    print()
    print("【4】横移速度 (m/s):", pct(dots))
    print(f"   >0.3: {100*np.mean(dots>0.3):.1f}%  "
          f">0.5: {100*np.mean(dots>0.5):.1f}%  "
          f">1.0: {100*np.mean(dots>1.0):.1f}%")
    print()
    print("=== 可信性自检 ===")
    ok_w = 5.0 < np.percentile(wd, 50) < 20.0
    ok_v = np.percentile(dots, 99) < 3.0
    print(f"   走廊宽度中位 {np.percentile(wd,50):.1f} m -> "
          f"{'合理' if ok_w else '异常'}")
    print(f"   横移速度 p99 {np.percentile(dots,99):.2f} m/s -> "
          f"{'合理' if ok_v else '异常（仍有参考跳变）'}")
    if ok_w and ok_v:
        if np.percentile(dl, 95) < 0.3:
            print("   => 数据里几乎没有横向机动")
        elif np.percentile(dl, 95) < 1.0:
            print(f"   => 有小幅横向机动（p95={np.percentile(dl,95):.2f} m）")
        else:
            print(f"   => 有明显横向机动（p95={np.percentile(dl,95):.2f} m）")

    if args.out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(args.out, exist_ok=True)
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
        axes[0, 0].hist(off, bins=100, color="#3182bd")
        axes[0, 0].set_xlabel("lateral offset to corridor center (m)")
        axes[0, 0].set_ylabel("count")
        axes[0, 0].set_title("all points (reference-locked)")
        axes[0, 0].grid(alpha=0.25)
        axes[0, 1].hist(dl, bins=60, color="#e6550d")
        axes[0, 1].axvline(0.5, color="#a50f15", ls="--", label="0.5 m")
        axes[0, 1].axvline(1.0, color="#54278f", ls=":", label="1.0 m")
        axes[0, 1].set_xlabel("|net lateral shift| (m)")
        axes[0, 1].set_title("per-trajectory lateral maneuver")
        axes[0, 1].legend(fontsize=8); axes[0, 1].grid(alpha=0.25)
        xs = np.sort(dl)
        axes[1, 0].plot(xs, np.linspace(0, 1, len(xs)), lw=2, color="#31a354")
        axes[1, 0].axvline(0.5, color="#a50f15", ls="--")
        axes[1, 0].set_xlabel("|net lateral shift| (m)")
        axes[1, 0].set_ylabel("CDF"); axes[1, 0].set_title("CDF")
        axes[1, 0].grid(alpha=0.25)
        for (w, tt, o) in ex[:4]:
            axes[1, 1].plot(tt, o, lw=1.6, label=f"world {w}")
        axes[1, 1].axhline(0, color="#999999", lw=0.8)
        axes[1, 1].set_xlabel("time (s)")
        axes[1, 1].set_ylabel("offset (m)")
        axes[1, 1].set_title("examples with |Δ|>1.0 m")
        axes[1, 1].legend(fontsize=7); axes[1, 1].grid(alpha=0.25)
        fig.tight_layout()
        fn = os.path.join(args.out, "lateral_offset.png")
        fig.savefig(fn, dpi=120); plt.close(fig)
        print("saved", fn)


if __name__ == "__main__":
    main()
