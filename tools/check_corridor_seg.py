"""证书可行性诊断（2/3）：用"点到线段"距离重算走廊。

修正点
------
1. `MapObservation` 是线段（position + length + heading），之前用点到点距离，
   会系统性高估净空。
2. type 0 的 length 恒为 0（实测 8901/10000 行），它是**点**不是线段；
   道路边界是 type 1（road_edge），另有 type 2（road_line）/ 3（lane_center）。
3. 张量是定长 10000 行，存在大量重复行（padding），画图/统计前先按
   (x, y, heading, length) 去重。

输出：专家路径自身净空、单位置最大可行横向偏移、各 offset 的越界比例、以及干净的渲染图。

用法（容器内）：
    python tools/check_corridor_seg.py --data-dir /data/data_json/training \
        --num-worlds 40 --seed 0 --out /workspace/data/plots/corridor_seg
"""
import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import numpy as np  # noqa: E402

IX_X, IX_Y, IX_LEN, IX_HEAD, IX_TYPE = 0, 1, 2, 5, 6
TYPE_COLORS = {1: "#e31a1c", 2: "#33a02c", 3: "#ff7f00"}


def dedup_rows(m):
    if not len(m):
        return m
    key = np.round(m[:, [IX_X, IX_Y, IX_LEN, IX_HEAD, IX_TYPE]], 4)
    _, idx = np.unique(key, axis=0, return_index=True)
    return m[np.sort(idx)]


def seg_endpoints(m):
    x, y, L, h = m[:, IX_X], m[:, IX_Y], m[:, IX_LEN], m[:, IX_HEAD]
    c, s = np.cos(h), np.sin(h)
    half = L / 2.0
    p0 = np.stack([x - half * c, y - half * s], axis=1)
    p1 = np.stack([x + half * c, y + half * s], axis=1)
    keep = L > 1e-3
    return p0[keep], p1[keep]


def dist_point_to_segments(q, a, b):
    """q [N,2] 到线段 a-b [M,2] 的最近距离 [N]。分块控制内存。"""
    out = np.full(len(q), np.inf)
    if len(a) == 0:
        return out
    ab = b - a                                    # [M,2]
    denom = np.maximum((ab ** 2).sum(1), 1e-12)   # [M]
    step = max(1, int(4e6 // max(1, len(a))))
    for i in range(0, len(q), step):
        qq = q[i:i + step]                        # [n,2]
        ap = qq[:, None, :] - a[None, :, :]       # [n,M,2]
        t = (ap * ab[None]).sum(-1) / denom[None]
        t = np.clip(t, 0.0, 1.0)
        proj = a[None] + t[..., None] * ab[None]
        d = np.linalg.norm(qq[:, None, :] - proj, axis=-1).min(axis=1)
        out[i:i + step] = d
    return out


def path_normals(p):
    d = np.gradient(p, axis=0)
    t = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    return np.stack([-t[:, 1], t[:, 0]], axis=1)


def max_offset(p, n, a, b, d_safe, sign, hi=5.0, step=0.25):
    best, o = 0.0, step
    while o <= hi + 1e-9:
        q = p + sign * o * n
        if dist_point_to_segments(q, a, b).min() < d_safe:
            break
        best, o = o, o + step
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--num-worlds", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d-safe", type=float, default=0.8)
    ap.add_argument("--types", default="1", help="作为边界使用的 type，逗号分隔")
    ap.add_argument("--offsets", default="0.5,1.0,1.2,1.5,2.0")
    ap.add_argument("--plot-worlds", type=int, default=4)
    ap.add_argument("--half-window", type=float, default=55.0)
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

    btypes = {int(t) for t in args.types.split(",")}
    offs = [float(x) for x in args.offsets.split(",")]
    rows, viol = [], {o: [] for o in offs}
    plots = []

    for w in range(args.num_worlds):
        m = dedup_rows(maps[w][np.abs(maps[w]).sum(1) > 1e-6])
        bnd = m[np.isin(m[:, IX_TYPE].astype(int), list(btypes))]
        a, b = seg_endpoints(bnd)
        if len(a) < 3:
            continue
        p = pos[w, ego[w]]
        ok = (np.abs(p[:, 0]) < 1e3) & (np.abs(p[:, 1]) < 1e3)
        if not ok.any():
            continue
        first_bad = int(np.argmin(ok)) if not ok.all() else len(ok)
        p = p[:first_bad]
        if len(p) < 10:
            continue
        n = path_normals(p)
        base = dist_point_to_segments(p, a, b)
        left = max_offset(p, n, a, b, args.d_safe, -1.0)
        right = max_offset(p, n, a, b, args.d_safe, +1.0)
        rows.append((w, base.min(), float(np.median(base)), left, right, len(bnd)))
        for o in offs:
            for sgn in (-1.0, +1.0):
                d = dist_point_to_segments(p + sgn * o * n, a, b)
                viol[o].append(float(np.mean(d < args.d_safe)))
        if len(plots) < args.plot_worlds:
            plots.append((w, m, p, n, base, left, right))

    if not rows:
        raise SystemExit("没有有效场景")

    R = np.asarray(rows, dtype=float)
    print(f"有效场景 {len(R)}/{args.num_worlds}  d_safe={args.d_safe} m  "
          f"边界 type={sorted(btypes)}")
    print(f"每场景边界线段数: p50={np.percentile(R[:,5],50):.0f} "
          f"p10={np.percentile(R[:,5],10):.0f}")
    print()
    print("专家路径自身净空（到边界线段）:")
    print(f"  最小净空: p50={np.percentile(R[:,1],50):.2f}  "
          f"p10={np.percentile(R[:,1],10):.2f}  min={R[:,1].min():.2f} m")
    print(f"  中位净空: p50={np.percentile(R[:,2],50):.2f} m")
    print()
    both = np.minimum(R[:, 3], R[:, 4])
    print("最大可行横向偏移（整条轨迹满足 d_safe）:")
    for pp in (10, 25, 50, 75, 90):
        print(f"  p{pp:<3d}: 左 {np.percentile(R[:,3],pp):.2f}  "
              f"右 {np.percentile(R[:,4],pp):.2f}  取小 {np.percentile(both,pp):.2f} m")
    for th in (0.5, 1.0, 1.2, 1.5, 2.0):
        print(f"  ±{th:.1f} m 完全可行的场景: {100*np.mean(both >= th):.0f}%")
    print()
    print("各 offset 越界点比例:")
    for o in offs:
        v = np.asarray(viol[o])
        print(f"  |o|={o:.1f} m: 越界点 {100*v.mean():5.1f}%  "
              f"整条无越界的场景 {100*np.mean(v <= 0):5.1f}%")

    if args.out and plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(args.out, exist_ok=True)
        for (w, m, p, n, base, left, right) in plots:
            fig, ax = plt.subplots(figsize=(7.6, 7.6))
            for t in np.unique(m[:, IX_TYPE].astype(int)):
                sel = m[m[:, IX_TYPE].astype(int) == t]
                a, b = seg_endpoints(sel)
                if len(a) == 0:
                    ax.scatter(sel[:, IX_X], sel[:, IX_Y], s=1.0, c="#dddddd",
                               linewidths=0, zorder=1)
                    continue
                # 同类线段用 NaN 分隔拼成一条折线，只产生一个图例句柄
                xs = np.empty(3 * len(a))
                ys = np.empty(3 * len(a))
                xs[0::3], xs[1::3], xs[2::3] = a[:, 0], b[:, 0], np.nan
                ys[0::3], ys[1::3], ys[2::3] = a[:, 1], b[:, 1], np.nan
                ax.plot(xs, ys, "-", lw=1.1,
                        c=TYPE_COLORS.get(t, "#c0c0c0"), alpha=0.85, zorder=2,
                        label=f"type {t} segments (n={len(a)})")
            ax.plot(p[:, 0], p[:, 1], "-", lw=2.6, c="#d01c8b", zorder=5,
                    label="expert path")
            for o, c in ((1.0, "#41ab5d"), (1.5, "#2171b5")):
                for sgn, ls in ((-1.0, "--"), (+1.0, ":")):
                    q = p + sgn * o * n
                    ax.plot(q[:, 0], q[:, 1], ls=ls, lw=1.2, c=c, alpha=0.9,
                            zorder=4, label=f"offset {o:.1f} m" if sgn < 0 else None)
            ax.set_aspect("equal")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7, loc="upper right", markerscale=4)
            ax.set_title(f"world {w}: expert min clearance={base.min():.2f} m | "
                         f"max offset L={left:.2f} R={right:.2f} m", fontsize=9)
            cx, cy = p[:, 0].mean(), p[:, 1].mean()
            r = args.half_window
            ax.set_xlim(cx - r, cx + r)
            ax.set_ylim(cy - r, cy + r)
            ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
            fig.tight_layout()
            fn = os.path.join(args.out, f"corridor_seg_w{w:02d}.png")
            fig.savefig(fn, dpi=115)
            plt.close(fig)
            print("saved", fn)


if __name__ == "__main__":
    main()
