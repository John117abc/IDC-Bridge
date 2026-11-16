"""验收 B+ 方案：从当前位姿 + 地图局部切向场生成前向路径与横向候选。

指标（与专家轨迹的比较只用于**评估**，不参与生成）：
  • 路径长度（应 ~90 m）
  • 与专家轨迹横向偏差（诊断用）
  • 相邻点最大跳变（应 ~1 m）
  • 航向突变（>30°/步 应为 0）
  • 每条横向候选的走廊可行性

用法（容器内）：
    python tools/check_map_candidates.py --data-dir /data/data_json/training \
        --num-worlds 16 --seed 0 --out /workspace/data/plots/bplus
"""
import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import numpy as np  # noqa: E402

IX_TYPE = 6
ROAD_EDGE = 1
OFFSETS = (-3.5, -1.6, -0.8, 0.0, 0.8, 1.6, 3.5)


def seg_dist(points, poly):
    a, b = poly[:-1], poly[1:]
    ab = b - a
    den = np.maximum((ab ** 2).sum(1), 1e-12)
    out = np.empty(len(points))
    for i, q in enumerate(points):
        t = np.clip(((a - q) * ab).sum(1) / den, 0.0, 1.0)
        out[i] = np.linalg.norm(q - (a + t[:, None] * ab), axis=1).min()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--config", default=os.path.join(_ROOT, "src", "configs", "eval.yaml"))
    ap.add_argument("--num-worlds", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot-worlds", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from gpudrive.env.dataset import SceneDataLoader
    from gpudrive.env.env_torch import GPUDriveTorchEnv

    from env.candidate_sampler import SamplerConfig
    from env.env_utils import get_ego_indices, get_env_config, load_scenes
    from env.idc_state_builder import GPUDriveObservationBuilder
    from env.map_candidates import LaneField, generate_map_candidates, integrate_path
    from utils.config import build_config

    cfg = build_config(args.config, cli_overrides={
        "num_worlds": args.num_worlds, "seed": args.seed, "device": "cuda",
        "gif_enabled": False})
    loader = SceneDataLoader(root=args.data_dir, batch_size=cfg.num_worlds,
                             dataset_size=cfg.num_worlds,
                             sample_with_replacement=False, shuffle=True,
                             seed=args.seed)
    env = GPUDriveTorchEnv(config=get_env_config(), data_loader=loader,
                           max_cont_agents=1, device=cfg.device,
                           action_type="continuous")
    scenes = load_scenes(env)
    builder = GPUDriveObservationBuilder(env, scenes)
    ego = get_ego_indices(env, cfg.num_worlds)
    maps = env.sim.map_observation_tensor().to_torch().cpu().numpy()
    expert = builder.expert_pos.detach().cpu().numpy()
    abs_np = env.sim.absolute_self_observation_tensor().to_torch().cpu().numpy()

    scfg = SamplerConfig()
    print(f"{'w':>3s} {'车道段':>6s} {'长度':>7s} {'专家长':>7s} {'偏差p50':>8s} "
          f"{'偏差p90':>8s} {'最大步':>7s} {'急转':>5s} {'可行候选':>8s}")
    rows, bases, cands_all = [], {}, {}
    for w in range(cfg.num_worlds):
        a = int(ego[w])
        pose = (float(abs_np[w, a, 0]), float(abs_np[w, a, 1]), float(abs_np[w, a, 7]))   # idx7 = heading
        speed0 = 8.0   # 真实管线里用 state[2]（因果的车速），这里用默认值
        field = LaneField(maps[w])
        cands, base = generate_map_candidates(pose, maps[w], OFFSETS, scfg,
                                             speed0=speed0)
        bases[w] = base
        cands_all[w] = cands
        L = np.linalg.norm(np.diff(base, axis=0), axis=1).sum()
        step = np.linalg.norm(np.diff(base, axis=0), axis=1)
        hd = np.unwrap(np.arctan2(np.diff(base[:, 1]), np.diff(base[:, 0])))
        sharp = int(np.sum(np.abs(np.diff(hd)) > np.deg2rad(30)))
        ex = expert[w, a]
        ex = ex[np.all(np.abs(ex) < 1e4, axis=1)]
        Lex = np.linalg.norm(np.diff(ex, axis=0), axis=1).sum() if len(ex) > 1 else np.nan
        if len(ex) > 2:
            d = seg_dist(ex, base)     # 专家点 -> 生成路径（专家短时更有意义）
            p50, p90 = float(np.median(d)), float(np.percentile(d, 90))
        else:
            p50 = p90 = np.nan
        n_ok = sum(1 for c in cands if c.cert["cor_ok"])
        vspd = np.mean([c.speed.mean() for c in cands])
        print(f"      参考速度均值={vspd:.1f} m/s  末速={np.mean([c.speed[-5:].mean() for c in cands]):.1f} m/s")
        rows.append((L, p50, step.max(), sharp, n_ok))
        print(f"{w:3d} {len(field):6d} {L:7.1f} {Lex:7.1f} {p50:8.2f} {p90:8.2f} "
              f"{step.max():7.2f} {sharp:5d} {n_ok:5d}/{len(cands)}")

    R = np.asarray(rows)
    print()
    print("=== 汇总（B+ 基准路径）===")
    print(f"  长度 p50={np.nanmedian(R[:,0]):.1f} m   (目标 ~91 m)")
    print(f"  与专家偏差 p50 中位={np.nanmedian(R[:,1]):.2f} m  "
          f"(<1.0 说明沿真实车道走)")
    print(f"  最大步长 p50={np.nanmedian(R[:,2]):.2f} m")
    print(f"  有急转的世界={int(np.sum(R[:,3]>0))}/{len(R)}")
    print(f"  可行候选数 p50={np.nanmedian(R[:,4]):.0f}/{len(OFFSETS)}")

    if args.out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(args.out, exist_ok=True)
        ws = list(range(min(args.plot_worlds, cfg.num_worlds)))
        n = len(ws)
        fig, axes = plt.subplots(1, n, figsize=(5.3 * n, 5.5), squeeze=False)
        for pi, w in enumerate(ws):
            ax = axes[0][pi]
            rows_m = maps[w][np.abs(maps[w]).sum(1) > 1e-6]
            bnd = rows_m[rows_m[:, IX_TYPE].astype(int) == ROAD_EDGE]
            if len(bnd):
                L, h = np.maximum(bnd[:, 2], 1e-3), bnd[:, 5]
                c, s = np.cos(h), np.sin(h)
                xs = np.full(3 * len(bnd), np.nan); ys = np.full(3 * len(bnd), np.nan)
                xs[0::3] = bnd[:, 0] - 0.5 * L * c
                xs[1::3] = bnd[:, 0] + 0.5 * L * c
                ys[0::3] = bnd[:, 1] - 0.5 * L * s
                ys[1::3] = bnd[:, 1] + 0.5 * L * s
                ax.plot(xs, ys, "-", lw=1.0, c="#bdbdbd", label="road edge")
            lane = rows_m[np.isin(rows_m[:, IX_TYPE].astype(int), (2, 3))]
            if len(lane):
                L2, h2 = np.maximum(lane[:, 2], 1e-3), lane[:, 5]
                c2, s2 = np.cos(h2), np.sin(h2)
                xs = np.full(3 * len(lane), np.nan); ys = np.full(3 * len(lane), np.nan)
                xs[0::3] = lane[:, 0] - 0.5 * L2 * c2
                xs[1::3] = lane[:, 0] + 0.5 * L2 * c2
                ys[0::3] = lane[:, 1] - 0.5 * L2 * s2
                ys[1::3] = lane[:, 1] + 0.5 * L2 * s2
                ax.plot(xs, ys, "-", lw=0.8, c="#9ecae1", alpha=0.8, label="lane seg")
            a = int(ego[w])
            ex = expert[w, a]
            ex = ex[np.all(np.abs(ex) < 1e4, axis=1)]
            ax.plot(ex[:, 0], ex[:, 1], "-", lw=2.0, c="#636363", label="expert (ref only)")
            cmap = plt.get_cmap("coolwarm")
            for i, c in enumerate(cands_all[w]):
                if not c.cert["cor_ok"]:
                    continue
                col = cmap((c.param["lat_offset"] + 3.5) / 7.0)
                ax.plot(c.pos[:, 0], c.pos[:, 1], "--", lw=1.3, c=col, alpha=0.9,
                        label=f"o={c.param['lat_offset']:+.1f}")
            ax.plot(bases[w][0, 0], bases[w][0, 1], "o", ms=7, c="#31a354")
            ax.set_aspect("equal"); ax.grid(alpha=0.22)
            ax.set_title(f"world {w}", fontsize=10)
            ax.legend(fontsize=6, loc="best")
            allp = np.concatenate([bases[w]] + [c.pos for c in cands_all[w]])
            cx, cy = allp[:, 0].mean(), allp[:, 1].mean()
            r = 0.6 * max(np.ptp(allp[:, 0]), np.ptp(allp[:, 1])) + 10
            ax.set_xlim(cx - r, cx + r); ax.set_ylim(cy - r, cy + r)
        fig.suptitle("B+ candidates from map tangent field (no expert future used)",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        fn = os.path.join(args.out, "bplus_candidates.png")
        fig.savefig(fn, dpi=125, bbox_inches="tight")
        print("saved", fn)


if __name__ == "__main__":
    main()
