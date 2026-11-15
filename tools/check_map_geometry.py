"""证书可行性诊断（1/3）：把 map_observation_tensor 按"线段"解包并渲染。

背景
----
`MapObservation` 的结构是 {position, scale(length,width), heading, type, id, mapType}，
即每个实体是**带长度和朝向的短线段**，不是点。之前的走廊判据用"点到点"距离，
会系统性高估净空；而且 type==1 只有 460/10000，疑似选错了子集。

本脚本回答：
  1. 张量里有多少行是**有效实体**（有效行数 vs 定长 10000）
  2. 每个 type 的实体数、线段长度分布
  3. 把实体画成线段后，道路结构是否连续（判断 position 是"中心"还是"端点"）

用法（容器内）：
    python tools/check_map_geometry.py --data-dir /data/data_json/training \
        --num-worlds 6 --seed 0 --out /workspace/data/plots/mapgeom
"""
import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import numpy as np  # noqa: E402

# 列含义（实测）：0=x 1=y 2=length 3=width 4=? 5=heading 6=type 7=id 8=mapType
IX_X, IX_Y, IX_LEN, IX_WID, IX_HEAD, IX_TYPE, IX_ID = 0, 1, 2, 3, 5, 6, 7

TYPE_COLORS = {
    0: "#1f78b4", 1: "#e31a1c", 2: "#33a02c",
    3: "#ff7f00", 4: "#6a3d9a", 5: "#b15928", 6: "#a6cee3",
}


def seg_endpoints(m, anchor="center"):
    """把每个实体转成线段的两端点。anchor: center（position 为中心）| start（为起点）"""
    x, y = m[:, IX_X], m[:, IX_Y]
    L = m[:, IX_LEN]
    h = m[:, IX_HEAD]
    c, s = np.cos(h), np.sin(h)
    if anchor == "center":
        half = L / 2.0
        p0 = np.stack([x - half * c, y - half * s], axis=1)
        p1 = np.stack([x + half * c, y + half * s], axis=1)
    else:
        p0 = np.stack([x, y], axis=1)
        p1 = np.stack([x + L * c, y + L * s], axis=1)
    return p0, p1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--num-worlds", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-plot", type=int, default=2)
    ap.add_argument("--half-window", type=float, default=60.0)
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

    print(f"map tensor shape: {maps.shape}")
    for w in range(min(args.num_worlds, 6)):
        m = maps[w]
        nz = np.abs(m).sum(1) > 1e-6
        rows = m[nz]
        # 有效行猜测：去掉与"上一行完全相同"的重复行（padding 常见做法）
        if len(rows):
            uniq = np.unique(np.round(rows[:, [IX_X, IX_Y, IX_HEAD, IX_LEN]], 4),
                             axis=0)
        else:
            uniq = rows
        types, cnt = np.unique(rows[:, IX_TYPE], return_counts=True) if len(rows) else ([], [])
        print(f"\nworld {w}: 非零行 {nz.sum()}/{len(m)}  去重后 {len(uniq)}")
        print(f"  type 分布: {dict(zip(np.round(types,0).astype(int).tolist(), cnt.tolist()))}")
        for t in np.unique(rows[:, IX_TYPE]) if len(rows) else []:
            L = rows[rows[:, IX_TYPE] == t][:, IX_LEN]
            if len(L) >= 3:
                print(f"    type {int(t):d}: n={len(L):5d}  "
                      f"len p50={np.percentile(L,50):6.2f} p95={np.percentile(L,95):6.2f} "
                      f"max={L.max():6.2f} m")
        p = pos[w, ego[w]]
        ok = (np.abs(p[:, 0]) < 1e3) & (np.abs(p[:, 1]) < 1e3)
        if ok.any():
            print(f"  expert path 范围: x[{p[ok,0].min():.0f},{p[ok,0].max():.0f}] "
                  f"y[{p[ok,1].min():.0f},{p[ok,1].max():.0f}]")

    # ---- 渲染：center vs start 两种假设 ----
    if args.out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(args.out, exist_ok=True)
        n = min(args.n_plot, args.num_worlds)
        fig, axes = plt.subplots(n, 2, figsize=(7.2 * 2, 6.4 * n), squeeze=False)
        for wi in range(n):
            w = wi
            p = pos[w, ego[w]]
            ok = (np.abs(p[:, 0]) < 1e3) & (np.abs(p[:, 1]) < 1e3)
            pv = p[ok]
            cx, cy = pv[:, 0].mean(), pv[:, 1].mean()
            for ci, anchor in enumerate(("center", "start")):
                ax = axes[wi][ci]
                rows = maps[w][np.abs(maps[w]).sum(1) > 1e-6]
                for t in np.unique(rows[:, IX_TYPE]):
                    sel = rows[rows[:, IX_TYPE] == t]
                    p0, p1 = seg_endpoints(sel, anchor=anchor)
                    big = sel[:, IX_LEN] > 0.3
                    if big.any():
                        ax.plot([p0[big, 0], p1[big, 0]], [p0[big, 1], p1[big, 1]],
                                "-", lw=1.0, c=TYPE_COLORS.get(int(t), "#888888"),
                                alpha=0.9, label=f"type {int(t)} (n={len(sel)})")
                    small = ~big
                    if small.any():
                        ax.plot(p0[small, 0], p0[small, 1], ".", ms=1.0,
                                c=TYPE_COLORS.get(int(t), "#888888"), alpha=0.5)
                if ok.any():
                    ax.plot(pv[:, 0], pv[:, 1], "-", lw=2.0, c="#d01c8b",
                            label="expert path", zorder=5)
                ax.set_xlim(cx - args.half_window, cx + args.half_window)
                ax.set_ylim(cy - args.half_window, cy + args.half_window)
                ax.set_aspect("equal")
                ax.grid(alpha=0.2)
                ax.set_title(f"world {w}  anchor={anchor}", fontsize=10)
                ax.legend(fontsize=6, loc="upper right", markerscale=4)
        fig.suptitle("MapObservation as segments: center vs start anchoring",
                     fontsize=12)
        fig.tight_layout()
        fn = os.path.join(args.out, "map_segments.png")
        fig.savefig(fn, dpi=110)
        plt.close(fig)
        print("\nsaved", fn)


if __name__ == "__main__":
    main()
