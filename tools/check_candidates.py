"""M0-b 验收：采样器 + 走廊裁剪 + 四判据证书，出图与统计。

用法（容器内）：
    python tools/check_candidates.py --data-dir /data/data_json/training \
        --num-worlds 24 --seed 0 --plot-worlds 4 \
        --out /workspace/data/plots/candidates
"""
import argparse
import os
import sys
from collections import Counter

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import numpy as np  # noqa: E402

from env.candidate_sampler import (                   # noqa: E402
    IX_TYPE, SamplerConfig, boundary_segments, build_candidates, curvature,
    dist_to_segments)

TYPE_COLORS = {1: "#9e9e9e", 2: "#33a02c", 3: "#ff7f00"}
CAND_COLORS = ["#d01c8b", "#2171b5", "#41ab5d", "#6a51a3", "#e6550d"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--num-worlds", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot-worlds", type=int, default=4)
    ap.add_argument("--half-window", type=float, default=45.0)
    ap.add_argument("--k-select", type=int, default=4)
    ap.add_argument("--delta", type=float, default=0.8)
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

    cfg = SamplerConfig(k_select=args.k_select, delta=args.delta)
    agg = Counter()
    clip_by_off = {}
    n_done = 0
    plots = []

    for w in range(args.num_worlds):
        p = pos[w, ego[w]]
        ok = (np.abs(p[:, 0]) < 1e3) & (np.abs(p[:, 1]) < 1e3)
        if not ok.any():
            continue
        first_bad = int(np.argmin(ok)) if not ok.all() else len(ok)
        exp = p[:first_bad]
        if len(exp) < 20:
            continue
        a, b = boundary_segments(maps[w])
        if len(a) < 5:
            continue
        selected, allc, infeasible = build_candidates(exp, maps[w], cfg)
        n_done += 1
        agg["raw"] += len(allc)
        agg["infeasible"] += len(infeasible)
        agg["selected"] += len(selected)
        for c in allc:
            k = c.param["lat_offset"]
            clip_by_off.setdefault(k, []).append(c.param["clipped_points"])
            if c.cert.get("traction_risk"):
                agg["traction_risk"] += 1
            for f in c.cert.get("kin_fail", []):
                agg[f"fail_{f}"] += 1
        for c in selected:
            agg[f"off{c.param['lat_offset']:+.1f}"] += 1
        if len(plots) < args.plot_worlds:
            plots.append((w, maps[w], exp, allc, selected, a, b))

        print(f"world {w:2d}: 边界线段 {len(a):4d}  专家净空 min="
              f"{dist_to_segments(exp, a, b).min():5.2f} m  "
              f"候选 {len(allc)} -> 可行 {len(allc)-len(infeasible)} -> "
              f"选中 {len(selected)}  "
              f"选中偏移 {[c.param['lat_offset'] for c in selected]}  "
              f"裁剪点 {[c.param['clipped_points'] for c in selected]}")

    print()
    print(f"=== 汇总（{n_done} 个场景）===")
    print(f"每场景候选 {agg['raw']/max(n_done,1):.1f} 条，"
          f"判 INFEASIBLE {agg['infeasible']/max(n_done,1):.1f} 条，"
          f"最终选中 {agg['selected']/max(n_done,1):.2f} 条")
    print("被选中的偏移分布:",
          {k: v for k, v in sorted(agg.items()) if k.startswith("off")})
    print("被判失败的原因（按候选条数统计）:",
          {k.replace("fail_", ""): v for k, v in sorted(agg.items())
           if k.startswith("fail_")})
    print(f"打 TRACTION_RISK 标记的候选: {agg['traction_risk']} / {agg['raw']} "
          f"({100*agg['traction_risk']/max(agg['raw'],1):.0f}%)")
    print("各偏移的平均裁剪点数:")
    for k in sorted(clip_by_off):
        v = np.asarray(clip_by_off[k])
        print(f"  {k:+.1f} m: 平均裁剪 {v.mean():5.1f} 点 / {T} 步 "
              f"({100*v.mean()/T:4.1f}%)  最大 {v.max()}")

    # ---- 图 ----
    if args.out and plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(args.out, exist_ok=True)
        for (w, wmap, exp, allc, selected, a, b) in plots:
            fig, ax = plt.subplots(figsize=(7.4, 7.4))
            # 道路（按 type 分组，NaN 拼接只留一个图例）
            rows = wmap[np.abs(wmap).sum(1) > 1e-6]
            for t in (1, 2, 3):
                sel = rows[rows[:, IX_TYPE].astype(int) == t]
                if not len(sel):
                    continue
                aa, bb = boundary_segments(sel, types=(t,))
                if not len(aa):
                    continue
                xs = np.empty(3 * len(aa)); ys = np.empty(3 * len(aa))
                xs[0::3], xs[1::3], xs[2::3] = aa[:, 0], bb[:, 0], np.nan
                ys[0::3], ys[1::3], ys[2::3] = aa[:, 1], bb[:, 1], np.nan
                ax.plot(xs, ys, "-", lw=1.0, c=TYPE_COLORS[t], alpha=0.8,
                        zorder=1, label=f"map type {t}")
            # 所有候选（细线）
            for c in allc:
                if c.status == "INFEASIBLE":
                    ax.plot(c.pos[:, 0], c.pos[:, 1], "-", lw=0.8, c="#d9d9d9",
                            zorder=2)
            # 选中候选（粗线）
            for i, c in enumerate(selected):
                ax.plot(c.pos[:, 0], c.pos[:, 1], "-", lw=2.2,
                        c=CAND_COLORS[i % len(CAND_COLORS)], zorder=4,
                        label=f"sel {c.param['lat_offset']:+.1f} m "
                              f"({c.status[10:14]})")
            ax.plot(exp[:, 0], exp[:, 1], "-", lw=1.6, c="#252525", ls="--",
                    zorder=3, label="expert")
            ax.set_aspect("equal"); ax.grid(alpha=0.2)
            ax.legend(fontsize=6.5, loc="upper right", markerscale=3)
            ax.set_title(f"world {w}: {len(allc)} raw -> {len(selected)} selected "
                         f"(k_select={cfg.k_select}, delta={cfg.delta} m)", fontsize=9)
            cx, cy = exp[:, 0].mean(), exp[:, 1].mean()
            r = args.half_window
            ax.set_xlim(cx - r, cx + r); ax.set_ylim(cy - r, cy + r)
            ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
            fig.tight_layout()
            fn = os.path.join(args.out, f"cand_w{w:02d}.png")
            fig.savefig(fn, dpi=115); plt.close(fig)
            print("saved", fn)

        # 净空剖面：展示裁剪把越界点拉回来
        w, wmap, exp, allc, selected, a, b = plots[0]
        fig, ax = plt.subplots(figsize=(8, 3.4))
        for i, c in enumerate(allc[::2]):
            ax.plot(c.clearance, lw=1.0, alpha=0.8,
                    label=f"off {c.param['lat_offset']:+.1f} (clip "
                          f"{c.param['clipped_points']})")
        ax.axhline(cfg.d_safe, color="#a50f15", ls="--", lw=1.2,
                   label=f"d_safe={cfg.d_safe}")
        ax.set_xlabel("step"); ax.set_ylabel("clearance (m)")
        ax.set_title(f"world {w}: clearance profile after corridor clipping",
                     fontsize=9)
        ax.grid(alpha=0.2); ax.legend(fontsize=6, ncol=2)
        fig.tight_layout()
        fn = os.path.join(args.out, "clearance_profile.png")
        fig.savefig(fn, dpi=115); plt.close(fig)
        print("saved", fn)


if __name__ == "__main__":
    main()
