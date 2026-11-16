"""画出 7 档横向候选（含 ±3.5 换道档）下，车分别跟每条时的实际轨迹。

修正前几版的两个问题：
  1. 只画"91 步全程存活"的世界（其余世界会提前终止，轨迹冻结会误导）
  2. 标签符号修正：自车系 +y = 左，所以 o>0 是"Left"

用法（容器内）：
    python tools/plot_candidates_run.py --data-dir /data/data_json/training \
        --model-path <pth> --num-worlds 8 --seed 0 --out /workspace/data/plots/cands7
"""
import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import numpy as np  # noqa: E402

IX_TYPE = 6
OFFSETS = (-3.5, -1.6, -0.8, 0.0, 0.8, 1.6, 3.5)


def label_of(o):
    if abs(o) < 1e-6:
        return "Center"
    side = "Left" if o > 0 else "Right"
    tag = " (lane change)" if abs(o) > 3.0 else ""
    return f"{side} {o:+.1f}{tag}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--config", default=os.path.join(_ROOT, "src", "configs", "eval.yaml"))
    ap.add_argument("--num-worlds", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot-worlds", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from gpudrive.env.dataset import SceneDataLoader
    from gpudrive.env.env_torch import GPUDriveTorchEnv

    from agents.idc_agent import DiscreteIDCAgent
    from env.candidate_sampler import SamplerConfig, install_fixed
    from env.env_utils import (extend_action_to_3d, get_ego_indices,
                               get_env_config, load_scenes)
    from env.idc_state_builder import GPUDriveObservationBuilder
    from utils.config import build_config

    cfg = build_config(args.config, cli_overrides={
        "num_worlds": args.num_worlds, "seed": args.seed, "device": "cuda",
        "model_path": args.model_path, "load_model": True, "gif_enabled": False})

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
    builder.generate_candidate_paths(ego, num_paths=1, use_expert_path=True)
    maps = env.sim.map_observation_tensor().to_torch().cpu().numpy()
    K = install_fixed(builder, ego, maps, OFFSETS)
    print(f"K={K} offsets={OFFSETS}")

    agent = DiscreteIDCAgent(env, cfg, cfg.device, builder, ego)
    agent.load(cfg.model_path)
    T = builder.EXPERT_TRAJ_LEN

    ref_start = agent.DIM_EGO + agent.DIM_OTHERS + agent.DIM_VALIDITY
    IX_LAT = ref_start + 3
    traj, valid, lats = {}, {}, {}
    for k in range(K):
        env.reset()
        rec, vrec, lrec = [], [], []
        for step in range(T):
            states = builder.get_idc_observations_batch(
                ego, path_indices=[k] * cfg.num_worlds)
            pos = np.array([[float(s[0]), float(s[1])] for s in states])
            rec.append(pos)
            vrec.append(np.all(np.abs(pos) < 1e3, axis=1))
            lrec.append(np.array([float(s[IX_LAT]) for s in states]))
            acts = agent.select_action(states, deterministic=True)
            env.step_dynamics(extend_action_to_3d(acts))
        traj[k] = np.asarray(rec)          # [T, W, 2]
        valid[k] = np.asarray(vrec)        # [T, W]
        lats[k] = np.asarray(lrec)         # [T, W]
        print(f"k={k} {label_of(OFFSETS[k]):22s} 全程存活世界数="
              f"{int(valid[k].all(0).sum())}/{cfg.num_worlds}")

    survive = np.ones(cfg.num_worlds, dtype=bool)
    for k in range(K):
        survive &= valid[k].all(0)
    survivors = [w for w in range(cfg.num_worlds) if survive[w]]
    print(f"所有候选都跑完 91 步的世界: {survivors}")

    print()
    print(f"{'candidate':26s} {'末段|lat|中位':>12s} {'末段|lat|p90':>12s} {'|o|':>6s}")
    for k in range(K):
        if not survivors:
            break
        tail = np.abs(lats[k][-20:, survivors])
        print(f"{label_of(OFFSETS[k]):26s} {np.median(tail):12.3f} "
              f"{np.percentile(tail, 90):12.3f} {abs(OFFSETS[k]):6.1f}")

    import matplotlib
    matplotlib.use("Agg")
    print()
    print(f"{'candidate':26s} {'末段|lat|中位':>12s} {'末段|lat|p90':>12s} {'|o|':>6s}")
    for k in range(K):
        if not survivors:
            break
        tail = np.abs(lats[k][-20:, survivors])
        print(f"{label_of(OFFSETS[k]):26s} {np.median(tail):12.3f} "
              f"{np.percentile(tail, 90):12.3f} {abs(OFFSETS[k]):6.1f}")

    import matplotlib.pyplot as plt

    os.makedirs(args.out, exist_ok=True)
    worlds = survivors[:args.plot_worlds] or list(range(min(2, cfg.num_worlds)))
    n = max(len(worlds), 1)
    fig, axes = plt.subplots(1, n, figsize=(5.4 * n, 5.6), squeeze=False)
    cmap = plt.get_cmap("coolwarm")
    colors = [cmap(i / max(K - 1, 1)) for i in range(K)]

    for pi, wi in enumerate(worlds):
        ax = axes[0][pi]
        a = int(ego[wi])
        rows = maps[wi][np.abs(maps[wi]).sum(1) > 1e-6]
        bnd = rows[rows[:, IX_TYPE].astype(int) == 1]
        if len(bnd):
            L, h = np.maximum(bnd[:, 2], 1e-3), bnd[:, 5]
            c, s = np.cos(h), np.sin(h)
            xs = np.full(3 * len(bnd), np.nan); ys = np.full(3 * len(bnd), np.nan)
            xs[0::3] = bnd[:, 0] - 0.5 * L * c
            xs[1::3] = bnd[:, 0] + 0.5 * L * c
            ys[0::3] = bnd[:, 1] - 0.5 * L * s
            ys[1::3] = bnd[:, 1] + 0.5 * L * s
            ax.plot(xs, ys, "-", lw=1.0, c="#bdbdbd", zorder=1)
        for k in range(K):
            p = builder.candidate_paths[wi][a][k]['pos']
            ax.plot(p[:, 0], p[:, 1], "--", lw=0.9, c=colors[k], alpha=0.5, zorder=2)
            tr = traj[k][:, wi, :]
            ax.plot(tr[:, 0], tr[:, 1], "-", lw=2.0, c=colors[k], zorder=4,
                    label=f"ego {label_of(OFFSETS[k])}")
        st = traj[K // 2][0, wi, :]
        ax.plot(st[0], st[1], "o", ms=7, c="#31a354", zorder=6)
        ax.set_aspect("equal"); ax.grid(alpha=0.22)
        ax.set_title(f"world {wi}", fontsize=10)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.legend(fontsize=6, loc="best", ncol=1)
        allp = np.concatenate([builder.candidate_paths[wi][a][k]['pos']
                               for k in range(K)])
        cx, cy = allp[:, 0].mean(), allp[:, 1].mean()
        r = 0.72 * max(np.ptp(allp[:, 0]), np.ptp(allp[:, 1])) + 14
        ax.set_xlim(cx - r, cx + r); ax.set_ylim(cy - r, cy + r)

    fig.suptitle(f"7 lateral candidates (o=0,+-0.8,+-1.6,+-3.5 m): dashed=reference, "
                 f"solid=ego actual", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fn = os.path.join(args.out, "candidates7.png")
    fig.savefig(fn, dpi=125, bbox_inches="tight")
    print("saved", fn)


if __name__ == "__main__":
    main()
