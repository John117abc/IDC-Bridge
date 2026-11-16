"""测量候选集对专家轨迹的覆盖率（决定"规划头"能不能学会的前提）。

对每个场景、每条由**位姿+地图**生成的候选，与专家轨迹做时间对齐比较：
  ADE_k   = mean_t |cand_k[t] - expert[t]|
  FDE_k   = |cand_k[T-1] - expert[T-1]|        （时间对齐，含速度误差）
  shape_k = mean_j dist(expert[j], cand_k 折线) （只看形状，忽略时间）
然后取 k 上的最小值，统计覆盖率。

判据：
  覆盖(FDE<2m)   ：候选集能否在"终点+速度"上表达专家
  覆盖(shape<1.5m)：候选集能否在"形状"上表达专家
  两者差距 → 说明失败主要来自速度/长度，而不是路线

用法（容器内）：
    python tools/measure_coverage.py --data-dir /data/data_json/training \
        --num-worlds 32 --batches 8 --out /workspace/data/plots/coverage
"""
import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import numpy as np  # noqa: E402

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
    ap.add_argument("--num-worlds", type=int, default=32)
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from gpudrive.env.dataset import SceneDataLoader
    from gpudrive.env.env_torch import GPUDriveTorchEnv

    from env.candidate_sampler import SamplerConfig
    from env.env_utils import get_ego_indices, get_env_config, load_scenes
    from env.idc_state_builder import GPUDriveObservationBuilder
    from env.map_candidates import generate_map_candidates
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
    scfg = SamplerConfig()

    rows = []          # (best_ade, best_fde, best_shape, best_off, ex_len, cand_len)
    for rep in range(args.batches):
        if rep > 0:
            env.reset()
            if hasattr(builder, '_setup_expert_data'):
                builder._setup_expert_data()
        maps = env.sim.map_observation_tensor().to_torch().cpu().numpy()
        expert = builder.expert_pos.detach().cpu().numpy()
        abs_np = env.sim.absolute_self_observation_tensor().to_torch().cpu().numpy()
        rel_np = env.sim.self_observation_tensor().to_torch().cpu().numpy()
        for w in range(cfg.num_worlds):
            a = int(ego[w])
            ex = expert[w, a]
            ok = np.all(np.abs(ex) < 1e4, axis=1)
            if ok.sum() < 50:
                continue
            ex = ex[ok]
            pose = (float(abs_np[w, a, 0]), float(abs_np[w, a, 1]),
                    float(abs_np[w, a, 7]))
            spd = float(np.clip(rel_np[w, a, 2], 1.0, 20.0))
            cands, _base = generate_map_candidates(pose, maps[w], OFFSETS, scfg,
                                                   speed0=spd)
            T = min(len(ex), 91)
            ades, fdes, shapes, offs = [], [], [], []
            for c in cands:
                p = np.asarray(c.pos[:T], dtype=np.float64)
                e = ex[:T]
                ades.append(float(np.linalg.norm(p - e, axis=1).mean()))
                fdes.append(float(np.linalg.norm(p[-1] - e[-1])))
                shapes.append(float(seg_dist(e, p).mean()))
                offs.append(c.param['lat_offset'])
            j = int(np.argmin(fdes))
            js = int(np.argmin(shapes))
            ex_len = float(np.linalg.norm(np.diff(ex, axis=0), axis=1).sum())
            cand_len = float(np.linalg.norm(np.diff(cands[0].pos, axis=0),
                                            axis=1).sum())
            # FDE 分解：沿专家末段航向的纵向误差 vs 横向误差
            u = e[-1] - e[-2]
            un = np.linalg.norm(u)
            u = u / max(un, 1e-6)
            nvec = np.array([-u[1], u[0]])
            err = np.asarray(cands[j].pos[-1], dtype=np.float64) - e[-1]
            rows.append((ades[j], fdes[j], shapes[js], offs[j], ex_len, cand_len,
                         abs(float(err @ u)), abs(float(err @ nvec))))

    if not rows:
        print("无有效样本")
        return
    R = np.asarray(rows)
    n = len(R)

    def pct(x, ps=(50, 75, 90, 95)):
        return "  ".join(f"p{p}={np.percentile(x, p):.2f}" for p in ps)

    print(f"\n样本数: {n}  (每个场景 {len(OFFSETS)} 条候选)")
    print()
    print("【FDE 分解（最优候选）】")
    print(f"  纵向(沿航向) |err|: p50={np.percentile(R[:,6],50):.2f}  "
          f"p90={np.percentile(R[:,6],90):.2f}")
    print(f"  横向(垂直)   |err|: p50={np.percentile(R[:,7],50):.2f}  "
          f"p90={np.percentile(R[:,7],90):.2f}")
    print()
    print("【最优候选】")
    print(f"  ADE   : {pct(R[:,0])}")
    print(f"  FDE   : {pct(R[:,1])}")
    print(f"  shape : {pct(R[:,2])}")
    print()
    print("【覆盖率】")
    for th in (1.0, 2.0, 3.0, 5.0):
        print(f"  FDE   < {th:.0f} m : {100*np.mean(R[:,1]<th):5.1f}%")
    for th in (0.5, 1.0, 1.5, 2.0):
        print(f"  shape < {th:.1f} m : {100*np.mean(R[:,2]<th):5.1f}%")
    print()
    print("【最优候选是不是中心(0)？】")
    print(f"  中心候选最优: {100*np.mean(np.abs(R[:,3])<1e-6):.1f}%   "
          f"需要横向偏移: {100*np.mean(np.abs(R[:,3])>1e-6):.1f}%")
    print()
    print("【长度对比：专家 vs 候选】")
    print(f"  专家  : {pct(R[:,4])}")
    print(f"  候选  : {pct(R[:,5])}")
    ratio = R[:,5] / np.maximum(R[:,4], 1e-6)
    print(f"  候选/专家 比值: {pct(ratio)}")
    print()
    # 失败归因
    bad_fde = R[:,1] >= 2.0
    bad_shape = R[:,2] >= 1.5
    print("【失败归因（FDE>=2 m 的样本）】")
    print(f"  共 {int(bad_fde.sum())} 个")
    if bad_fde.sum():
        print(f"  其中形状也差(shape>=1.5m): {int((bad_fde & bad_shape).sum())}"
              f"  → 路线错")
        print(f"  其中形状尚可(shape<1.5m) : {int((bad_fde & ~bad_shape).sum())}"
              f"  → **速度/长度错**")

    if args.out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(args.out, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
        axes[0].hist(R[:,1], bins=40, color="#3182bd")
        axes[0].axvline(2.0, color="#a50f15", ls="--")
        axes[0].set_title("best FDE (time-aligned)"); axes[0].grid(alpha=.25)
        axes[1].hist(R[:,2], bins=40, color="#41ab5d")
        axes[1].axvline(1.5, color="#a50f15", ls="--")
        axes[1].set_title("best shape error"); axes[1].grid(alpha=.25)
        axes[2].scatter(R[:,4], R[:,5], s=8, alpha=.5, c="#6a51a3")
        lim = max(R[:,4].max(), R[:,5].max())
        axes[2].plot([0, lim], [0, lim], "k--", lw=1)
        axes[2].set_xlabel("expert length (m)"); axes[2].set_ylabel("candidate length (m)")
        axes[2].set_title("length: expert vs candidate"); axes[2].grid(alpha=.25)
        fig.suptitle(f"candidate coverage vs expert (n={n})", fontweight="bold")
        fig.tight_layout()
        fn = os.path.join(args.out, "coverage.png")
        fig.savefig(fn, dpi=120, bbox_inches="tight")
        print("saved", fn)


if __name__ == "__main__":
    main()
