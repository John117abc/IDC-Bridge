"""画出"三条候选 + 车分别跟这三条"的轨迹对比（效果图）。

同一场景、同一批初始状态，只改"全部世界强制走第 k 条候选"：
  候选集固定为 (−0.8, 0.0, +0.8)，索引 0/1/2 分别对应 左/中/右。
每种情况跑满 91 步，记录 ego 实际轨迹，叠加画在同一张图上。

用法（容器内）：
    python tools/plot_ab_trajectories.py --data-dir /data/data_json/training \
        --model-path <pth> --num-worlds 8 --seed 0 \
        --out /workspace/data/plots/ab_traj
"""
import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402
sys.path.insert(0, os.path.join(_ROOT, "src"))

COLORS = {0: "#2171b5", 1: "#252525", 2: "#de2d26"}
NAMES = {0: "Left -0.8 m", 1: "Center", 2: "Right +0.8 m"}
IX_TYPE = 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--config", default=os.path.join(_ROOT, "src", "configs", "eval.yaml"))
    ap.add_argument("--num-worlds", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot-worlds", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch

    from gpudrive.env.dataset import SceneDataLoader
    from gpudrive.env.env_torch import GPUDriveTorchEnv

    from agents.idc_agent import DiscreteIDCAgent
    from env.candidate_sampler import install_fixed
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
    K = install_fixed(builder, ego, maps, (-0.8, 0.0, 0.8))
    print(f"候选数 K={K}，offsets=(-0.8, 0.0, +0.8)")

    agent = DiscreteIDCAgent(env, cfg, cfg.device, builder, ego)
    agent.load(cfg.model_path)
    T = builder.EXPERT_TRAJ_LEN

    # 记录每个候选索引下的 ego 轨迹
    traj = {}      # idx -> [T, W, 2]
    for idx in range(K):
        env.reset()
        rec = []
        for step in range(T):
            states = builder.get_idc_observations_batch(
                ego, path_indices=[idx] * cfg.num_worlds)
            rec.append(np.array([[float(s[0]), float(s[1])] for s in states]))
            acts = agent.select_action(states, deterministic=True)
            env.step_dynamics(extend_action_to_3d(acts))
        traj[idx] = np.asarray(rec)          # [T, W, 2]
        print(f"idx={idx} ({NAMES[idx]}) 已跑完 {T} 步")

    # 画图
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(args.out, exist_ok=True)
    n = min(args.plot_worlds, cfg.num_worlds)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5.4), squeeze=False)
    for wi in range(n):
        ax = axes[0][wi]
        a = int(ego[wi])
        rows = maps[wi][np.abs(maps[wi]).sum(1) > 1e-6]
        bnd = rows[rows[:, IX_TYPE].astype(int) == 1]
        if len(bnd):
            L, h = np.maximum(bnd[:, 2], 1e-3), bnd[:, 5]
            c, s = np.cos(h), np.sin(h)
            xs = np.empty(3 * len(bnd)); ys = np.empty(3 * len(bnd))
            xs[0::3] = bnd[:, 0] - 0.5 * L * c
            xs[1::3] = bnd[:, 0] + 0.5 * L * c
            xs[2::3] = np.nan
            ys[0::3] = bnd[:, 1] - 0.5 * L * s
            ys[1::3] = bnd[:, 1] + 0.5 * L * s
            ys[2::3] = np.nan
            ax.plot(xs, ys, "-", lw=0.9, c="#bdbdbd", zorder=1,
                    label="road boundary (type1)")
        # 候选参考
        for k in range(K):
            p = builder.candidate_paths[wi][a][k]['pos']
            ax.plot(p[:, 0], p[:, 1], "--", lw=1.2, c=COLORS[k], alpha=0.55,
                    zorder=2)
        # 实际轨迹
        for k in range(K):
            tr = traj[k][:, wi, :]
            ok = np.abs(tr[:, 0]) < 1e3
            ax.plot(tr[ok, 0], tr[ok, 1], "-", lw=2.4, c=COLORS[k], zorder=4,
                    label=f"ego actual ({NAMES[k]})")
        ax.plot(traj[1][0, wi, 0], traj[1][0, wi, 1], "o", ms=7, c="#31a354",
                zorder=6, label="same start")
        ax.set_aspect("equal"); ax.grid(alpha=0.22)
        ax.legend(fontsize=7, loc="best")
        ax.set_title(f"world {wi}", fontsize=10)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        # 以三条候选的并集定视野
        allp = np.concatenate([builder.candidate_paths[wi][a][k]['pos']
                               for k in range(K)])
        cx, cy = allp[:, 0].mean(), allp[:, 1].mean()
        r = 1.25 * max(np.ptp(allp[:, 0]), np.ptp(allp[:, 1])) / 2 + 12
        ax.set_xlim(cx - r, cx + r); ax.set_ylim(cy - r, cy + r)

    fig.suptitle("Same scene, same start: 3 candidates (dashed) vs ego trajectories (solid)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fn = os.path.join(args.out, "ab_trajectories.png")
    fig.savefig(fn, dpi=125, bbox_inches="tight")
    print("saved", fn)


if __name__ == "__main__":
    main()
