"""M0-a：实测专家轨迹的曲率 / 侧向加速度分布，用于给候选采样定界。

## 为什么要实测

转向限幅 `δ ∈ [-0.6, 0.6] rad`、轴距 `L ≈ 5.286 m` 给出的运动学上限是
`κ_max = tan(0.6)/L = 0.129 1/m`（R_min ≈ 7.7 m），在 15 m/s 时对应
`a_lat = v²κ ≈ 29 m/s²` —— 完全不现实。**真正约束候选的是人类驾驶实际的
曲率/舒适度分布**，所以从数据里量，而不是用转向限幅。

## 估计量的选择（两版都不完美，记录清楚）

1. **二阶差分**（首版）：在停驻点/重复点上 `speed→0`，κ 爆到 800+ 1/m，弃用。
2. **航向差分 / 弧长**（当前采用）：`κ = Δheading/Δs`，先平滑位置、unwrap 航向、
   过滤 `Δs < 0.15 m` 的段、在首个哨兵点处截断轨迹。可解释，但**低速段有噪声**：
   实测 p5 段长仅 0.19 m，0.02 rad 的航向抖动就给出 κ≈0.1，与运动学上限同量级，
   因此"13.9% 轨迹超出运动学上限"是噪声尾巴，不是真实曲率。
3. **滑动窗圆拟合**（试过，弃用）：近直线的 7 点窗是退化问题（点集近似共线），
   半径解不稳定、R² 可能为负，clamp 后 κ 爆到 1e3~1e6，比差分更差。

## 因此采用"双规则"定界

- **曲率硬上限**：`κ ≤ κ_cand = 0.25 1/m`（专家单条最大曲率的 p99，偏保守）
- **侧向加速度上限**（主规则，按速度自洽）：
  `κ(s) ≤ a_lat_cand / v(s)²`，取 `a_lat_cand = 3.0 m/s²`（专家单条最大值 p95）
  例：v=10 m/s → κ≤0.03 (R≥33 m)；v=5 m/s → κ≤0.12 (R≥8.3 m)

用法（容器内）：
    python tools/stat_curvature.py --data-dir /data/data_json/training \
        --num-worlds 75 --seed 0 --out /workspace/data/plots/curvature.png
"""
import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import numpy as np  # noqa: E402


def moving_avg_xy(p, k=3):
    if k <= 1 or len(p) < k:
        return p
    ker = np.ones(k) / k
    out = np.empty_like(p)
    for d in range(p.shape[1]):
        pad = np.pad(p[:, d], (k // 2, k - 1 - k // 2), mode="edge")
        out[:, d] = np.convolve(pad, ker, mode="valid")
    return out


def curvature_by_heading(p, dt, ds_min=0.15):
    """κ = Δheading/Δs。返回 (kappa, speed, ds)，长度比 p 少 2。"""
    ps = moving_avg_xy(p, 3)
    d = np.diff(ps, axis=0)
    ds = np.linalg.norm(d, axis=1)
    heading = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    dhead = np.diff(heading)
    ds_mid = 0.5 * (ds[1:] + ds[:-1])
    ok = ds_mid > ds_min
    kappa = dhead[ok] / ds_mid[ok]
    speed = (ds / dt)[:-1][ok]
    return kappa, speed, ds_mid[ok]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--num-worlds", type=int, default=75)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--wheelbase", type=float, default=5.286)
    ap.add_argument("--steer-max", type=float, default=0.6)
    ap.add_argument("--a-lat-max", type=float, default=3.0,
                    help="侧向加速度上限 (m/s^2)，专家单条最大值 p95")
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

    raw = env.sim.expert_trajectory_tensor().to_torch().cpu().numpy()
    W, A, _ = raw.shape
    T = 91
    pos = raw[:, :, :2 * T].reshape(W, A, T, 2)

    K_all, V_all, ds_all, kmax, amax = [], [], [], [], []
    n_short = 0
    for w in range(W):
        p = pos[w, ego[w]]
        valid = (np.abs(p[:, 0]) < 1e3) & (np.abs(p[:, 1]) < 1e3)
        if not valid.any():
            n_short += 1
            continue
        first_bad = int(np.argmin(valid)) if not valid.all() else len(valid)
        p = p[:first_bad]
        if len(p) < 8:
            n_short += 1
            continue
        kappa, speed, ds = curvature_by_heading(p, args.dt)
        if len(kappa) < 5:
            n_short += 1
            continue
        K_all.append(kappa)
        V_all.append(speed)
        ds_all.append(ds)
        kmax.append(np.abs(kappa).max())
        amax.append((np.abs(speed) ** 2 * np.abs(kappa)).max())

    if not K_all:
        raise SystemExit("没有有效轨迹")

    kappa = np.abs(np.concatenate(K_all))
    speed = np.concatenate(V_all)
    ds = np.concatenate(ds_all)
    kmax = np.abs(np.asarray(kmax))
    amax = np.asarray(amax)
    alat = np.abs(speed) ** 2 * np.abs(kappa)

    kappa_veh = np.tan(args.steer_max) / args.wheelbase

    def pct(a, ps=(50, 90, 95, 99)):
        return "  ".join(f"p{p}={np.percentile(a, p):.4f}" for p in ps)

    print(f"有效轨迹 {len(kmax)}/{W}（跳过 {n_short} 条过短/无效）")
    print(f"曲率段 {len(kappa)} 个；段长 ds: p5={np.percentile(ds,5):.3f} "
          f"p50={np.percentile(ds,50):.3f} max={ds.max():.3f} m")
    print(f"速度 (m/s)      : p50={np.percentile(speed,50):.2f}  "
          f"p95={np.percentile(speed,95):.2f}  max={speed.max():.2f}")
    print(f"曲率 |kappa|    : {pct(kappa)}")
    print(f"  -> 半径 (m)   : p50={1/max(np.percentile(kappa,50),1e-9):.1f}  "
          f"p95={1/max(np.percentile(kappa,95),1e-9):.1f}  "
          f"p99={1/max(np.percentile(kappa,99),1e-9):.1f}")
    print(f"单条轨迹最大曲率: {pct(kmax)}")
    print(f"侧向加速度     : {pct(alat)}")
    print(f"单条轨迹最大侧向加速度: p50={np.percentile(amax,50):.2f}  "
          f"p95={np.percentile(amax,95):.2f}  p99={np.percentile(amax,99):.2f}")
    print()
    print(f"车辆运动学上限   : tan({args.steer_max})/{args.wheelbase} "
          f"= {kappa_veh:.4f} 1/m (R_min={1/kappa_veh:.1f} m)")
    print(f"超出运动学上限的轨迹: {100*np.mean(kmax > kappa_veh):.1f}%"
          f"  <- 噪声尾巴（低速段 ds 小），不代表真实曲率")

    k_cand = float(np.percentile(kmax, 99))
    print()
    print("=== 候选生成定界（采用双规则）===")
    print(f"1) 曲率硬上限   : kappa_cand = {k_cand:.4f} 1/m "
          f"(R={1/max(k_cand,1e-9):.1f} m)  = 专家单条最大曲率的 p99")
    print(f"2) 侧向加速度   : a_lat_cand = {args.a_lat_max:.2f} m/s^2 "
          f"(专家 p95={np.percentile(amax,95):.2f})")
    print("   -> 逐点规则 κ(s) ≤ min(kappa_cand, a_lat_cand / v(s)^2)")
    for v in (5.0, 10.0, 15.0, 20.0):
        print(f"      v={v:4.1f} m/s -> κ ≤ {min(k_cand, args.a_lat_max/v**2):.4f} "
              f"(R ≥ {1/min(k_cand, args.a_lat_max/v**2):.1f} m)")

    if args.out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
        axes[0].hist(kappa, bins=80, color="#3182bd")
        axes[0].axvline(k_cand, color="#e6550d", ls="--",
                        label=f"kappa_cand={k_cand:.3f}")
        axes[0].axvline(kappa_veh, color="#a50f15", ls=":",
                        label=f"kinematic={kappa_veh:.3f}")
        axes[0].set_yscale("log")
        axes[0].set_xlabel("curvature (1/m)")
        axes[0].set_ylabel("count")
        axes[0].legend(fontsize=7)
        axes[0].set_title("expert curvature (per segment)")
        axes[1].hist(alat, bins=80, color="#31a354")
        axes[1].axvline(args.a_lat_max, color="#e6550d", ls="--",
                        label=f"a_lat_cand={args.a_lat_max}")
        axes[1].set_yscale("log")
        axes[1].set_xlabel("lateral accel v^2*|kappa| (m/s^2)")
        axes[1].legend(fontsize=7)
        axes[1].set_title("implied lateral acceleration")
        fig.tight_layout()
        fig.savefig(args.out, dpi=120)
        print("saved", args.out)


if __name__ == "__main__":
    main()
