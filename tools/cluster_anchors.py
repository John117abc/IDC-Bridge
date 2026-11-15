"""本地：对采集到的专家轨迹做 K-Means 聚类，得到 anchor 集并出图。

输入：tools/collect_expert_traj.py 产出的 traj_seed*.npz（自车系轨迹）
输出：anchor 集（簇均值）、簇大小、簇内横向离散度，以及若干张图

关键诊断：
  - 簇大小分布  -> K 是否过大
  - 簇内 y 的标准差 -> 该簇是不是"同一路线的不同横向偏移"
  - 簇均值的末端横向偏移 y(T) -> 数据里到底有没有横向模态
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np                # noqa: E402


def kmeans(X, k, iters=60, n_init=5, seed=0):
    """朴素 K-Means（k-means++ 初始化）。X: [N, D]"""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        # k-means++
        centers = [X[rng.integers(len(X))]]
        d2 = ((X - centers[0]) ** 2).sum(1)
        for _ in range(k - 1):
            p = d2 / d2.sum()
            c = X[rng.choice(len(X), p=p)]
            centers.append(c)
            d2 = np.minimum(d2, ((X - c) ** 2).sum(1))
        C = np.asarray(centers)
        for _ in range(iters):
            d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
            lab = d.argmin(1)
            newC = np.asarray([X[lab == j].mean(0) if np.any(lab == j) else C[j]
                               for j in range(k)])
            if np.allclose(newC, C, atol=1e-4):
                C = newC
                break
            C = newC
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1)
        inertia = float(d[np.arange(len(X)), lab].sum())
        if best is None or inertia < best[0]:
            best = (inertia, C, lab)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="npz 目录")
    ap.add_argument("--out", default="plots/anchors")
    ap.add_argument("--ks", default="8,16,32")
    ap.add_argument("--stride", type=int, default=5, help="时间下采样步长")
    ap.add_argument("--lat-weight", type=float, default=1.0,
                    help="y（横向）在距离度量里的权重，>1 更看重横向差异")
    ap.add_argument("--min-length", type=float, default=0.0,
                    help="只对里程 >= 该值的轨迹聚类（静止场景单独统计）")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "traj_seed*.npz")))
    if not files:
        raise SystemExit(f"{args.dir} 下没有 traj_seed*.npz")
    T_all, L_all = [], []
    for f in files:
        d = np.load(f)
        T_all.append(d["traj"])
        L_all.append(d["length"])
    traj = np.concatenate(T_all)          # [N, 91, 2]
    length = np.concatenate(L_all)
    N0 = len(traj)
    if args.min_length > 0:
        move = length >= args.min_length
        print(f"里程过滤 >= {args.min_length:.0f} m: 保留 {move.sum()}/{N0} "
              f"({100*move.mean():.0f}%)；剔除的 {N0-move.sum()} 条"
              f"（里程 p50={np.percentile(length[~move],50) if (~move).any() else 0:.1f} m）"
              f"视为'静止/极低速'场景")
        traj, length = traj[move], length[move]
    N, T, _ = traj.shape
    print(f"载入 {N} 条轨迹（{len(files)} 个文件），时域 {T} 步 = {T*0.1:.1f} s")
    print(f"里程: p10={np.percentile(length,10):.1f} p50={np.percentile(length,50):.1f} "
          f"p90={np.percentile(length,90):.1f} m")

    idx = np.arange(0, T, args.stride)
    pts = traj[:, idx, :]                 # [N, n_pts, 2]
    w = np.array([1.0, args.lat_weight])
    X = (pts * w).reshape(N, -1)          # [N, n_pts*2]
    print(f"特征维度 {X.shape[1]}（{len(idx)} 个时刻 × 2）")

    os.makedirs(args.out, exist_ok=True)
    ks = [int(x) for x in args.ks.split(",")]
    results = {}
    for k in ks:
        inertia, C, lab = kmeans(X, k, seed=0)
        C = C.reshape(k, len(idx), 2) / w
        results[k] = (inertia, C, lab)
        sizes = np.bincount(lab, minlength=k)
        order = np.argsort(-sizes)
        print(f"\n===== K={k}  inertia={inertia:.0f} =====")
        print("簇大小（降序）:", " ".join(f"{sizes[i]}({100*sizes[i]/N:.0f}%)"
                                     for i in order[:12]))
        print(f"最小簇占比 {100*sizes.min()/N:.1f}%   空簇 {int((sizes==0).sum())}")
        # 每簇的末端纵向/横向
        print("每簇 末端 (x, y) 与簇内 y 标准差:")
        for j in order[:12]:
            m = lab == j
            if not m.any():
                continue
            y_end = C[j, -1, 1]
            y_std = float(traj[m][:, -1, 1].std())
            x_end = C[j, -1, 0]
            print(f"  c{j:2d} n={sizes[j]:4d}  x_end={x_end:6.1f} m  "
                  f"y_end={y_end:+5.2f} m  簇内y_std={y_std:4.2f} m")

    # ---- 画图：K 最小和中等各一张 ----
    for k in [ks[0], ks[min(1, len(ks) - 1)]]:
        inertia, C, lab = results[k]
        sizes = np.bincount(lab, minlength=k)
        order = np.argsort(-sizes)
        ncol = 4
        nrow = int(np.ceil(k / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.8 * nrow),
                                 squeeze=False)
        for pos, j in enumerate(order):
            ax = axes[pos // ncol][pos % ncol]
            m = lab == j
            sub = traj[m][:, idx, :]
            for s in sub[::max(1, len(sub) // 60)]:
                ax.plot(s[:, 0], s[:, 1], "-", lw=0.4, c="#bdbdbd", alpha=0.6)
            ax.plot(C[j][:, 0], C[j][:, 1], "-", lw=2.2, c="#d01c8b")
            ax.plot(C[j][0, 0], C[j][0, 1], "o", ms=4, c="#2ca02c")
            ax.plot(C[j][-1, 0], C[j][-1, 1], "s", ms=4, c="#d62728")
            ax.set_title(f"c{j}  n={sizes[j]} ({100*sizes[j]/N:.0f}%)\n"
                         f"y_end={C[j,-1,1]:+.1f} m", fontsize=7)
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(alpha=0.2)
            ax.tick_params(labelsize=6)
        for pos in range(k, nrow * ncol):
            axes[pos // ncol][pos % ncol].axis("off")
        fig.suptitle(f"Anchors from K-Means over {N} expert trajectories "
                     f"(ego frame, K={k})", fontsize=11)
        fig.tight_layout()
        fn = os.path.join(args.out, f"anchors_K{k:02d}.png")
        fig.savefig(fn, dpi=110)
        plt.close(fig)
        print("saved", fn)

    # ---- 横向模态诊断：所有簇末端横向偏移的分布 ----
    inertia, C, lab = results[ks[0]]
    y_end = C[:, -1, 1]
    print(f"\n[横向模态诊断] K={ks[0]} 各 anchor 末端横向偏移 y_end: "
          f"{np.round(np.sort(y_end), 2).tolist()}")
    print(f"  跨度 {y_end.max()-y_end.min():.2f} m；"
          f"|y_end|>1.5m 的 anchor 占比 {100*np.mean(np.abs(y_end) > 1.5):.0f}%")


if __name__ == "__main__":
    main()
