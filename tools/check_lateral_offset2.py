"""Q1-v2：用**走廊中点**作参考，测数据里有没有横向行为。

为什么换参考
------------
v1 用"最近的 lane_center 分段"作参考，结果被严重污染：最近分段会在相邻平行车道
和曲线处跳变，跳一次就掺进 3.5 m 的假位移，导致"净横向位移 p50=3.65 m、
车道 id 切换 100%"这种不可能的数字。

v2 改用走廊中点：对每个 ego 点，取 type-1 边界线段中**左侧最近**和**右侧最近**的
距离 d_left / d_right（按 ego 航向的带符号投影判定左右），则
    offset_mid = (d_right - d_left) / 2        # + 表示偏右
这个量只依赖边界几何，不受车道分段 id 影响。

判读标准
--------
如果 |Δoffset_mid| 的 p95 < 0.3 m，且横移速度分布集中在 0.1 m/s 以下，
说明数据里几乎没有横向机动（大家都贴中线开），横向候选的价值就有限。

用法（容器内）：
    python tools/check_lateral_offset2.py --data-dir /data/data_json/training \
        --num-worlds 40 --seed 0 --out /workspace/data/plots/lateral2
"""
import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import numpy as np  # noqa: E402

IX_X, IX_Y, IX_LEN, IX_HEAD, IX_TYPE = 0, 1, 2, 5, 6
ROAD_EDGE = 1


def boundary_segments(rows):
    m = rows[rows[:, IX_TYPE].astype(int) == ROAD_EDGE]
    if not len(m):
        return None
    key = np.round(m[:, [IX_X, IX_Y, IX_LEN, IX_HEAD]], 4)
    _, idx = np.unique(key, axis=0, return_index=True)
    m = m[np.sort(idx)]
    x, y, L, h = m[:, IX_X], m[:, IX_Y], np.maximum(m[:, IX_LEN], 1e-3), m[:, IX_HEAD]
    c, s = np.cos(h), np.sin(h)
    a = np.stack([x - 0.5 * L * c, y - 0.5 * L * s], 1)
    b = np.stack([x + 0.5 * L * c, y + 0.5 * L * s], 1)
    return a, b


def nearest_side_dist(p, h, a, b, side):
    """side=+1 取左侧最近距离，-1 取右侧。用带符号投影判定左右。"""
    ab = b - a
    den = np.maximum((ab ** 2).sum(1), 1e-12)
    t = np.clip(((p - a) * ab).sum(1) / den, 0.0, 1.0)
    proj = a + t[:, None] * ab
    d = np.linalg.norm(p - proj, axis=1)
    # 用投影点的相对位置判断在 ego 的左还是右
    rel = proj - p
    n = np.array([-np.sin(h), np.cos(h)])       # ego 左法向
    sgn = rel @ n
    sel = (sgn * side) > 0.2                    # 留 0.2 m 死区
    if not sel.any():
        return np.inf
    return float(d[sel].min())


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
        a, b = seg
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

        off = np.full(len(p), np.nan)
        width = np.full(len(p), np.nan)
        for i in range(len(p)):
            dl = nearest_side_dist(p[i], hs[i], a, b, +1.0)
            dr = nearest_side_dist(p[i], hs[i], a, b, -1.0)
            if np.isfinite(dl) and np.isfinite(dr):
                off[i] = 0.5 * (dr - dl)
                width[i] = dl + dr
        v = np.isfinite(off)
        if v.sum() < 20:
            continue
        n_scene += 1
        o = off[v]
        all_off.append(o)
        delta = float(o[-1] - o[0])
        per.append(dict(w=w, max_abs=float(np.abs(o).max()),
                        delta=abs(delta), swing=float(o.max() - o.min()),
                        width=float(np.nanmedian(width[v]))))
        if len(ex) < args.plot_n and abs(delta) > 1.2:
            ex.append((w, np.arange(v.sum()) * 0.1, o))

    if not per:
        raise SystemExit("没有有效场景")

    off = np.concatenate(all_off)
    A = np.abs(off)
    mx = np.asarray([r["max_abs"] for r in per])
    dl = np.asarray([r["delta"] for r in per])
    sw = np.asarray([r["swing"] for r in per])
    wd = np.asarray([r["width"] for r in per])

    def pct(x, ps=(50, 75, 90, 95, 99)):
        return "  ".join(f"p{p}={np.percentile(x, p):.2f}" for p in ps)

    print(f"有效场景 {n_scene}，采样点 {len(off)}")
    print(f"走廊宽度（左右边界之和）: p50={np.percentile(wd,50):.1f} m")
    print()
    print("【1】相对走廊中点的横向偏移 |offset| (m):")
    print(f"   {pct(A)}")
    print(f"   >0.5 m: {100*np.mean(A>0.5):.1f}%   >1.0 m: "
          f"{100*np.mean(A>1.0):.1f}%   >1.5 m: {100*np.mean(A>1.5):.1f}%")
    print()
    print("【2】单条轨迹 (m):")
    print(f"   最大|offset|: {pct(mx)}")
    print(f"   净横向位移|Δ|: {pct(dl)}")
    print(f"   横向摆动幅度:  {pct(sw)}")
    print()
    print("【3】横向机动事件:")
    for th in (0.2, 0.3, 0.5, 1.0, 1.5):
        print(f"   |Δoffset| > {th:.1f} m 的轨迹: {100*np.mean(dl > th):5.1f}%")
    print()
    dots = [np.abs(np.diff(o)) / 0.1 for o in all_off if len(o) > 5]
    dots = np.concatenate(dots) if dots else np.array([0.0])
    print("【4】横移速度 |d offset/dt| (m/s):")
    print(f"   {pct(dots)}")
    print(f"   >0.3 m/s: {100*np.mean(dots>0.3):.1f}%   >0.5 m/s: "
          f"{100*np.mean(dots>0.5):.1f}%   >1.0 m/s: {100*np.mean(dots>1.0):.1f}%")
    print()
    print("=== 判定 ===")
    if np.percentile(dl, 95) < 0.3:
        print("  数据里几乎没有横向机动（p95 < 0.3 m）→ 横向候选价值有限")
    elif np.percentile(dl, 95) < 1.0:
        print("  有小幅横向机动（0.3~1.0 m）→ 横向候选只对'小幅避让'有意义")
    else:
        print(f"  有明显横向机动（p95 = {np.percentile(dl,95):.2f} m）→ 值得学")

    if args.out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(args.out, exist_ok=True)
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
        axes[0, 0].hist(off, bins=100, color="#3182bd")
        axes[0, 0].set_xlabel("lateral offset to corridor center (m)")
        axes[0, 0].set_ylabel("count")
        axes[0, 0].set_title("all sampled points"); axes[0, 0].grid(alpha=0.25)
        axes[0, 1].hist(dl, bins=60, color="#e6550d")
        axes[0, 1].axvline(0.5, color="#a50f15", ls="--", label="0.5 m")
        axes[0, 1].axvline(1.0, color="#54278f", ls=":", label="1.0 m")
        axes[0, 1].set_xlabel("|net lateral shift| per trajectory (m)")
        axes[0, 1].set_title("lateral maneuver magnitude")
        axes[0, 1].legend(fontsize=8); axes[0, 1].grid(alpha=0.25)
        xs = np.sort(dl)
        axes[1, 0].plot(xs, np.linspace(0, 1, len(xs)), lw=2, color="#31a354")
        axes[1, 0].axvline(0.5, color="#a50f15", ls="--")
        axes[1, 0].set_xlabel("|net lateral shift| (m)")
        axes[1, 0].set_ylabel("CDF")
        axes[1, 0].set_title("CDF"); axes[1, 0].grid(alpha=0.25)
        for (w, tt, o) in ex[:4]:
            axes[1, 1].plot(tt, o, lw=1.6, label=f"world {w}")
        axes[1, 1].axhline(0, color="#999999", lw=0.8)
        axes[1, 1].set_xlabel("time (s)")
        axes[1, 1].set_ylabel("offset to corridor center (m)")
        axes[1, 1].set_title("examples with |Δ|>1.2 m")
        axes[1, 1].legend(fontsize=7); axes[1, 1].grid(alpha=0.25)
        fig.tight_layout()
        fn = os.path.join(args.out, "lateral_offset.png")
        fig.savefig(fn, dpi=120); plt.close(fig)
        print("saved", fn)


if __name__ == "__main__":
    main()
