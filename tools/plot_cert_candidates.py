"""带证书（走廊 + 差异性）的候选集：统计 K_eff + 跟踪误差 + 出图。

与 install_fixed 的区别：这里每条候选都要过走廊证书（clearance >= d_safe）
和 Hausdorff 差异性判据，不可行的会被丢掉 —— 这正是之前"左侧车不动"的
场景应该被过滤掉的地方。

用法（容器内）：
    python tools/plot_cert_candidates.py --data-dir /data/data_json/training \
        --model-path <pth> --num-worlds 32 --seed 0 --out /workspace/data/plots/cert
"""
import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import numpy as np  # noqa: E402

IX_TYPE = 6
ALL_OFFSETS = (-3.5, -1.6, -0.8, 0.0, 0.8, 1.6, 3.5)


def label_of(o):
    if abs(o) < 1e-6:
        return "Center"
    return f"{'Left ' if o > 0 else 'Right'} {o:+.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--config", default=os.path.join(_ROOT, "src", "configs", "eval.yaml"))
    ap.add_argument("--num-worlds", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot-worlds", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from gpudrive.env.dataset import SceneDataLoader
    from gpudrive.env.env_torch import GPUDriveTorchEnv

    from agents.idc_agent import DiscreteIDCAgent
    from env.candidate_sampler import (SamplerConfig, boundary_segments,
                                       build_candidates, make_candidate)
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

    scfg = SamplerConfig(k_select=5)
    experts = builder.expert_pos.detach().cpu().numpy()

    # ---- 逐 world 用证书筛候选 ----
    per_world_offsets = {}
    n_infeasible = 0
    for w in range(cfg.num_worlds):
        a = int(ego[w])
        a_seg, b_seg = boundary_segments(maps[w])
        sel, raw, inf = build_candidates(experts[w, a], maps[w], scfg)
        n_infeasible += len(inf)
        if not sel:
            sel = [make_candidate(experts[w, a], 0.0, 0.0, a_seg, b_seg, scfg)]
        paths = []
        for c in sel:
            paths.append({
                'pos': c.pos.astype(np.float32),
                'heading': c.heading.astype(np.float32),
                'speed': c.speed.astype(np.float32),
                'road_dist': np.maximum(c.clearance, 0.0).astype(np.float32),
            })
        per_world_offsets[w] = [c.param['lat_offset'] for c in sel]
        builder.candidate_paths[w] = {a: paths}

    K = max(len(v) for v in per_world_offsets.values())
    for w, d in builder.candidate_paths.items():
        a = list(d.keys())[0]
        while len(d[a]) < K:
            d[a].append(dict(d[a][-1]))
            per_world_offsets[w].append(per_world_offsets[w][-1])
    builder.num_candidate_paths = K
    builder.rebuild_paths_tensor()

    ks = np.array([len(set(v)) for v in per_world_offsets.values()])
    print("=" * 84)
    print(f"证书筛选结果: K_max={K}, 每 world 可用候选数 中位={np.median(ks):.0f} "
          f"min={ks.min()} max={ks.max()}, 累计 INFEASIBLE={n_infeasible}")
    has = {o: sum(1 for w in range(cfg.num_worlds) if o in per_world_offsets[w])
           for o in ALL_OFFSETS}
    print("各档位可用率: " + "  ".join(
        f"{label_of(o)}={has[o]}/{cfg.num_worlds}" for o in ALL_OFFSETS))

    # ---- 逐档位跑闭环，记录 lat 与轨迹 ----
    agent = DiscreteIDCAgent(env, cfg, cfg.device, builder, ego)
    agent.load(cfg.model_path)
    ref_start = agent.DIM_EGO + agent.DIM_OTHERS + agent.DIM_VALIDITY
    IX_LAT = ref_start + 3
    T = builder.EXPERT_TRAJ_LEN

    lat_by_o, traj_by_o, valid_by_o = {}, {}, {}
    for o in ALL_OFFSETS:
        idx = [per_world_offsets[w].index(o) if o in per_world_offsets[w]
               else 0 for w in range(cfg.num_worlds)]
        has_w = np.array([o in per_world_offsets[w] for w in range(cfg.num_worlds)])
        env.reset()
        lrec, prec, vrec = [], [], []
        for step in range(T):
            states = builder.get_idc_observations_batch(ego, path_indices=idx)
            pos = np.array([[float(s[0]), float(s[1])] for s in states])
            lrec.append(np.array([float(s[IX_LAT]) for s in states]))
            prec.append(pos)
            vrec.append(np.all(np.abs(pos) < 1e3, axis=1))
            acts = agent.select_action(states, deterministic=True)
            env.step_dynamics(extend_action_to_3d(acts))
        lat_by_o[o] = np.asarray(lrec)
        traj_by_o[o] = np.asarray(prec)
        valid_by_o[o] = np.asarray(vrec)
        tail = np.abs(lat_by_o[o][-20:][:, has_w])
        print(f"  {label_of(o):12s} 覆盖 {has_w.sum():2d} 世界  "
              f"末段|lat|中位={np.median(tail):6.3f}  p90={np.percentile(tail,90):7.3f}")

    # ---- 出图 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(args.out, exist_ok=True)
    # 选"候选数最少"和"候选数最多"的世界各一两个，体现证书的作用
    order = sorted(range(cfg.num_worlds),
                   key=lambda w: (len(set(per_world_offsets[w])), w))
    worlds = order[:1] + order[-args.plot_worlds + 1:] if args.plot_worlds > 1 else order[:1]
    n = len(worlds)
    fig, axes = plt.subplots(1, n, figsize=(5.6 * n, 5.8), squeeze=False)
    cmap = plt.get_cmap("coolwarm")
    cmap_by_o = {o: cmap((o + 3.5) / 7.0) for o in ALL_OFFSETS}

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
        offs_here = sorted(set(per_world_offsets[wi]))
        for o in offs_here:
            k = per_world_offsets[wi].index(o)
            p = builder.candidate_paths[wi][a][k]['pos']
            ax.plot(p[:, 0], p[:, 1], "--", lw=0.9, c=cmap_by_o[o], alpha=0.55, zorder=2)
            tr = traj_by_o[o][:, wi, :]
            ok = valid_by_o[o][:, wi]
            ax.plot(tr[ok, 0], tr[ok, 1], "-", lw=2.2, c=cmap_by_o[o], zorder=4,
                    label=f"ego {label_of(o)}")
        st = traj_by_o[0.0][0, wi, :]
        ax.plot(st[0], st[1], "o", ms=7, c="#31a354", zorder=6)
        ax.set_aspect("equal"); ax.grid(alpha=0.22)
        ax.set_title(f"world {wi}  (K_eff={len(offs_here)})", fontsize=10)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.legend(fontsize=6.5, loc="best")
        allp = np.concatenate([builder.candidate_paths[wi][a][k]['pos']
                               for k in range(len(builder.candidate_paths[wi][a]))])
        cx, cy = allp[:, 0].mean(), allp[:, 1].mean()
        r = 0.72 * max(np.ptp(allp[:, 0]), np.ptp(allp[:, 1])) + 14
        ax.set_xlim(cx - r, cx + r); ax.set_ylim(cy - r, cy + r)

    fig.suptitle("CERTIFIED candidates only (corridor + Hausdorff filter): "
                 "dashed=reference, solid=ego actual", fontsize=12,
                 fontweight="bold")
    fig.tight_layout()
    fn = os.path.join(args.out, "certified_candidates.png")
    fig.savefig(fn, dpi=125, bbox_inches="tight")
    print("saved", fn)


if __name__ == "__main__":
    main()
