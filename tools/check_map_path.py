"""排查 A 方案：现有地图路径生成器（use_expert_path=False）到底有没有问题。

用户回忆症状 = "最终弄出来的道路很混乱"，这里把它量化：

  1. 路径长度：能不能拼出 80~90 m
  2. 与专家轨迹的横向偏差：是不是沿真实行驶路线
  3. 跳变：相邻点最大间距（正常应 <= 2 m @ 0.1 s）
  4. 折返/混乱：航向突变次数、曲率异常点数
  5. 视觉：4 个场景的地图边界 + 专家轨迹 + 生成路径叠画

用法（容器内）：
    python tools/check_map_path.py --data-dir /data/data_json/training \
        --num-worlds 16 --seed 0 --out /workspace/data/plots/mappath
"""
import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import numpy as np  # noqa: E402

IX_TYPE = 6
ROAD_EDGE = 1


def seg_dist(points, poly):
    """每个点到折线 poly 的最近距离。"""
    a, b = poly[:-1], poly[1:]
    ab = b - a
    den = np.maximum((ab ** 2).sum(1), 1e-12)
    out = np.empty(len(points))
    for i, q in enumerate(points):
        t = np.clip(((a - q) * ab).sum(1) / den, 0.0, 1.0)
        proj = a + t[:, None] * ab
        out[i] = np.linalg.norm(q - proj, axis=1).min()
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

    from env.env_utils import get_ego_indices, get_env_config, load_scenes
    from env.idc_state_builder import GPUDriveObservationBuilder
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

    print("生成地图路径（use_expert_path=False）...")
    builder.generate_candidate_paths(ego, num_paths=1, use_expert_path=False)

    print()
    print(f"{'w':>3s} {'长度':>7s} {'专家长':>7s} {'偏差p50':>8s} {'偏差p90':>8s} "
          f"{'最大跳':>7s} {'急转':>5s} {'覆盖率':>7s}")
    rows = []
    paths = {}
    for w in range(cfg.num_worlds):
        a = int(ego[w])
        try:
            p = np.asarray(builder.candidate_paths[w][a][0]['pos'], dtype=np.float64)
        except Exception as e:                       # noqa: BLE001
            print(f"{w:3d} 生成失败: {e}")
            continue
        ok = np.all(np.abs(p) < 1e4, axis=1)
        p = p[ok]
        if len(p) < 5:
            print(f"{w:3d} 有效点太少 ({len(p)})")
            continue
        paths[w] = p
        ex = expert[w, a]
        ex_ok = np.all(np.abs(ex) < 1e4, axis=1)
        ex = ex[ex_ok]

        L = np.linalg.norm(np.diff(p, axis=0), axis=1).sum()
        Lex = np.linalg.norm(np.diff(ex, axis=0), axis=1).sum()
        if len(ex) > 1:
            d = seg_dist(p, ex)
            p50, p90 = np.median(d), np.percentile(d, 90)
            cover = float(np.mean(d < 3.0))
        else:
            p50 = p90 = cover = np.nan
        step = np.linalg.norm(np.diff(p, axis=0), axis=1)
        maxjump = float(step.max())
        hd = np.unwrap(np.arctan2(np.diff(p[:, 1]), np.diff(p[:, 0])))
        dh = np.abs(np.diff(hd)) if len(hd) > 1 else np.array([0.0])
        sharp = int(np.sum(dh > np.deg2rad(30)))
        rows.append((L, p50, maxjump, sharp, cover))
        print(f"{w:3d} {L:7.1f} {Lex:7.1f} {p50:8.2f} {p90:8.2f} "
              f"{maxjump:7.2f} {sharp:5d} {cover*100:6.0f}%")

    if rows:
        R = np.asarray(rows, dtype=np.float64)
        print()
        print("=== 汇总 ===")
        print(f"  路径长度:  p50={np.nanmedian(R[:,0]):.1f} m  "
              f"≥80m 占比={100*np.mean(R[:,0]>=80):.0f}%")
        print(f"  横向偏差:  p50 中位={np.nanmedian(R[:,1]):.2f} m")
        print(f"  最大跳变:  p50={np.nanmedian(R[:,2]):.2f} m  "
              f">5m 的世界={int(np.sum(R[:,2]>5))}/{len(R)}")
        print(f"  急转(>30°): 有急转的世界={int(np.sum(R[:,3]>0))}/{len(R)}")
        print(f"  覆盖率(3m内): p50={np.nanmedian(R[:,4])*100:.0f}%")

    # ---- 出图 ----
    if args.out and paths:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(args.out, exist_ok=True)
        ws = sorted(paths.keys())[:args.plot_worlds]
        n = max(len(ws), 1)
        fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5.4), squeeze=False)
        for pi, w in enumerate(ws):
            ax = axes[0][pi]
            rows_m = maps[w][np.abs(maps[w]).sum(1) > 1e-6]
            bnd = rows_m[rows_m[:, IX_TYPE].astype(int) == ROAD_EDGE]
            if len(bnd):
                L, h = np.maximum(bnd[:, 2], 1e-3), bnd[:, 5]
                c, s = np.cos(h), np.sin(h)
                xs = np.full(3 * len(bnd), np.nan)
                ys = np.full(3 * len(bnd), np.nan)
                xs[0::3] = bnd[:, 0] - 0.5 * L * c
                xs[1::3] = bnd[:, 0] + 0.5 * L * c
                ys[0::3] = bnd[:, 1] - 0.5 * L * s
                ys[1::3] = bnd[:, 1] + 0.5 * L * s
                ax.plot(xs, ys, "-", lw=1.0, c="#bdbdbd", label="road edge")
            a = int(ego[w])
            ex = expert[w, a]
            ex = ex[np.all(np.abs(ex) < 1e4, axis=1)]
            ax.plot(ex[:, 0], ex[:, 1], "-", lw=2.2, c="#636363",
                    label="expert (GT future)")
            p = paths[w]
            ax.plot(p[:, 0], p[:, 1], "-", lw=2.2, c="#de2d26",
                    label="map-generated path")
            ax.plot(p[0, 0], p[0, 1], "o", ms=7, c="#31a354")
            ax.set_aspect("equal"); ax.grid(alpha=0.22)
            ax.set_title(f"world {w}", fontsize=10)
            ax.legend(fontsize=7)
            allp = np.concatenate([ex, p])
            cx, cy = allp[:, 0].mean(), allp[:, 1].mean()
            r = 0.62 * max(np.ptp(allp[:, 0]), np.ptp(allp[:, 1])) + 12
            ax.set_xlim(cx - r, cx + r); ax.set_ylim(cy - r, cy + r)
        fig.suptitle("A-scheme check: map-generated path (red) vs expert future (grey)",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        fn = os.path.join(args.out, "map_path_check.png")
        fig.savefig(fn, dpi=125, bbox_inches="tight")
        print("saved", fn)


if __name__ == "__main__":
    main()
