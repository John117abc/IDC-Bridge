"""M0-诊断：走廊（road corridor）到底有多宽？±1.2 m 的横向偏移可行吗？

背景：`road_dist` 在现有代码里就是"到 road_edge 点云（type==1）的最近距离"
（`IDCStateBuilder._min_dist_to_pts`）。所以"走廊约束"可以直接用现成数据量化，
不需要新的传感器或地图。

本脚本回答三个问题：
  1. 每条横向偏移轨迹上，有多少比例的点 road_dist < d_safe（压出走廊）
  2. 每个场景**最大可行的横向偏移**是多少（左右分别二分/扫描）
  3. 直接画出来看：专家路径、±offset 偏移、以及越界点

用法（容器内）：
    python tools/check_corridor.py --data-dir /data/data_json/training \
        --num-worlds 40 --seed 0 --out /workspace/data/plots/corridor
"""
import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import numpy as np  # noqa: E402

ROAD_EDGE = 1


def min_dist_to_pts(pts, query):
    """query [N,2] 到 pts [M,2] 的最近距离 [N]（分块避免爆内存）。"""
    out = np.empty(len(query), dtype=np.float64)
    step = max(1, int(2e6 // max(1, len(pts))))
    for i in range(0, len(query), step):
        q = query[i:i + step]
        d2 = ((pts[None, :, 0] - q[:, None, 0]) ** 2
              + (pts[None, :, 1] - q[:, None, 1]) ** 2)
        out[i:i + step] = np.sqrt(d2.min(axis=1))
    return out


def path_normals(p):
    """逐点单位法向量（切线左转 90°）。"""
    d = np.gradient(p, axis=0)
    t = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    n = np.stack([-t[:, 1], t[:, 0]], axis=1)
    return n


def max_feasible_offset(expert, normals, edges, d_safe, sign, hi=4.0, step=0.25):
    """沿 sign 方向逐步增大偏移，返回仍然全部满足 road_dist>=d_safe 的最大值。"""
    best = 0.0
    o = step
    while o <= hi + 1e-9:
        q = expert + sign * o * normals
        d = min_dist_to_pts(edges, q)
        if d.min() < d_safe:
            break
        best = o
        o += step
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--num-worlds", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d-safe", type=float, default=0.8,
                    help="走廊安全距离 (m)")
    ap.add_argument("--offsets", default="0.5,1.0,1.2,1.5,2.0,3.0")
    ap.add_argument("--plot-worlds", type=int, default=4)
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
    W, A, _ = raw.shape
    T = 91
    pos = raw[:, :, :2 * T].reshape(W, A, T, 2)

    offs = [float(x) for x in args.offsets.split(",")]

    # ---- 统计 ----
    rows = []
    viol_by_off = {o: [] for o in offs}
    plot_pool = []

    for w in range(W):
        road = maps[w]
        edges = road[road[:, 6] == ROAD_EDGE][:, :2]
        if len(edges) < 10:
            continue
        p = pos[w, ego[w]]
        valid = (np.abs(p[:, 0]) < 1e3) & (np.abs(p[:, 1]) < 1e3)
        if valid.sum() < 10:
            continue
        first_bad = int(np.argmin(valid)) if not valid.all() else len(valid)
        p = p[:first_bad]
        if len(p) < 10:
            continue
        n = path_normals(p)

        base = min_dist_to_pts(edges, p)
        left = max_feasible_offset(p, n, edges, args.d_safe, -1.0)
        right = max_feasible_offset(p, n, edges, args.d_safe, +1.0)
        rows.append((w, len(p), float(base.min()), float(np.median(base)),
                     left, right))

        for o in offs:
            for sgn in (-1.0, +1.0):
                q = p + sgn * o * n
                d = min_dist_to_pts(edges, q)
                viol_by_off[o].append(float(np.mean(d < args.d_safe)))

        if len(plot_pool) < args.plot_worlds:
            plot_pool.append((w, p, n, edges, base, left, right))

    if not rows:
        raise SystemExit("没有有效场景")

    R = np.asarray(rows, dtype=float)
    print(f"有效场景 {len(R)}/{W}   d_safe={args.d_safe} m")
    print()
    print("专家路径自身的 road_dist:")
    print(f"  最小值 p50={np.percentile(R[:,2],50):.2f}  "
          f"p10={np.percentile(R[:,2],10):.2f}  min={R[:,2].min():.2f} m")
    print(f"  中位数 p50={np.percentile(R[:,3],50):.2f} m （越大越居中）")
    print()
    print("最大可行横向偏移（整条轨迹都满足 road_dist>=d_safe）:")
    both = np.minimum(R[:, 4], R[:, 5])
    for p in (10, 25, 50, 75, 90):
        print(f"  p{p:<3d}: 左侧 {np.percentile(R[:,4],p):.2f} m   "
              f"右侧 {np.percentile(R[:,5],p):.2f} m   取小 {np.percentile(both,p):.2f} m")
    for th in (1.0, 1.2, 1.5, 2.0):
        ok = np.mean(both >= th) * 100
        print(f"  ±{th:.1f} m 完全可行的场景比例: {ok:.0f}%")
    print()
    print("各 offset 下越界点比例（所有场景平均）:")
    for o in offs:
        v = np.asarray(viol_by_off[o])
        if len(v):
            print(f"  |offset|={o:.1f} m: 越界点 {100*v.mean():5.1f}%   "
                  f"整条完全无越界的场景 {100*np.mean(v <= 0):5.1f}%")

    # ---- 画图 ----
    if args.out and plot_pool:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(args.out, exist_ok=True)
        for (w, p, n, edges, base, left, right) in plot_pool:
            fig, ax = plt.subplots(figsize=(7.2, 7.2))
            ax.scatter(edges[:, 0], edges[:, 1], s=1.2, c="#8c510a",
                       linewidths=0, label="road_edge (type=1)", zorder=1)
            ax.plot(p[:, 0], p[:, 1], "-", lw=2.6, c="#d01c8b",
                    label="expert path", zorder=4)
            colors = {1.0: "#41ab5d", 1.5: "#2171b5", 2.0: "#6a51a3"}
            for o, c in colors.items():
                for sgn, ls in ((-1.0, "--"), (+1.0, ":")):
                    q = p + sgn * o * n
                    d = min_dist_to_pts(edges, q)
                    bad = d < args.d_safe
                    ax.plot(q[:, 0], q[:, 1], ls=ls, lw=1.3,
                            c=c, alpha=0.85,
                            label=f"offset {'-' if sgn < 0 else '+'}{o:.1f} m"
                                  if ls == "--" or sgn > 0 else None, zorder=3)
                    if bad.any():
                        ax.scatter(q[bad, 0], q[bad, 1], marker="x", s=26,
                                   c="#a50f15", zorder=5,
                                   label="violates d_safe" if (o == 1.0 and sgn < 0) else None)
            ax.set_aspect("equal")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7, loc="best")
            ax.set_title(f"world {w}: expert road_dist min={base.min():.2f} m, "
                         f"max feasible offset  left={left:.2f}  right={right:.2f} m",
                         fontsize=9)
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            # 视野限制在专家路径附近
            cx, cy = p[:, 0].mean(), p[:, 1].mean()
            r = max(30.0, 1.15 * max(np.ptp(p[:, 0]), np.ptp(p[:, 1])))
            ax.set_xlim(cx - r, cx + r)
            ax.set_ylim(cy - r, cy + r)
            fig.tight_layout()
            fn = os.path.join(args.out, f"corridor_world{w:02d}.png")
            fig.savefig(fn, dpi=115)
            plt.close(fig)
            print("saved", fn)


if __name__ == "__main__":
    main()
