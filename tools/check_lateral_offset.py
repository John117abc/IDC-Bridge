"""Q1：数据里到底有没有"横向行为"？

用 lane_center（type 3）作车道参考，算 ego 专家轨迹相对**最近且同向**车道中心线的
带符号横向偏移，回答：
  1. 横向偏移的绝对量级分布（大部分时间是不是就贴中心线开）
  2. 单条轨迹的**净横向位移** |Δoffset| 和**横向摆动幅度** max-min
  3. 最近车道 id 是否发生切换（= 换道事件）
  4. 横向偏移随时间的变化率（横移速度）

判定标准：如果 |Δoffset| 的 p95 < 0.3 m 且几乎从不换道，说明**数据里没有值得学的
横向行为**，横向候选的价值只剩"绕障碍"，且样本极稀疏。

用法（容器内）：
    python tools/check_lateral_offset.py --data-dir /data/data_json/training \
        --num-worlds 40 --seed 0 --out /workspace/data/plots/lateral
"""
import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import numpy as np  # noqa: E402

IX_X, IX_Y, IX_LEN, IX_HEAD, IX_TYPE, IX_ID = 0, 1, 2, 5, 6, 7
LANE_CENTER = 3
ALIGN_COS = np.cos(np.deg2rad(60.0))     # 同向判定阈值


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def lane_segments(rows):
    m = rows[rows[:, IX_TYPE].astype(int) == LANE_CENTER]
    if not len(m):
        return None
    key = np.round(m[:, [IX_X, IX_Y, IX_LEN, IX_HEAD, IX_ID]], 4)
    _, idx = np.unique(key, axis=0, return_index=True)
    m = m[np.sort(idx)]
    x, y, L, h, i = m[:, IX_X], m[:, IX_Y], m[:, IX_LEN], m[:, IX_HEAD], m[:, IX_ID]
    L = np.maximum(L, 1e-3)
    c, s = np.cos(h), np.sin(h)
    a = np.stack([x - 0.5 * L * c, y - 0.5 * L * s], 1)
    b = np.stack([x + 0.5 * L * c, y + 0.5 * L * s], 1)
    return {"a": a, "b": b, "h": h, "id": i}


def project(p, h, lanes):
    """返回 (signed_lateral_offset, segment_id, seg_heading) 或 (nan, -1, nan)。"""
    a, b, sh, sid = lanes["a"], lanes["b"], lanes["h"], lanes["id"]
    cosd = np.cos(wrap(sh - h))
    ok = cosd > ALIGN_COS
    if not ok.any():
        return np.nan, -1, np.nan
    aa, bb = a[ok], b[ok]
    ab = bb - aa
    den = np.maximum((ab ** 2).sum(1), 1e-12)
    t = np.clip(((p - aa) * ab).sum(1) / den, 0.0, 1.0)
    proj = aa + t[:, None] * ab
    d = np.linalg.norm(p - proj, axis=1)
    j = int(np.argmin(d))
    hh = sh[ok][j]
    n = np.array([-np.sin(hh), np.cos(hh)])      # 车道左法向
    s = float(np.dot(p - aa[j], n))
    return s, int(sid[ok][j]), float(hh)


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

    all_off, per_traj = [], []      # per_traj: (max|off|, |Δoff|, swing, n_ids, dur)
    examples = []
    n_scene = 0

    for w in range(args.num_worlds):
        rows = maps[w][np.abs(maps[w]).sum(1) > 1e-6]
        lanes = lane_segments(rows)
        if lanes is None or len(lanes["a"]) < 3:
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
        offs, ids = [], []
        for i in range(len(p)):
            s, sid, _ = project(p[i], hs[i], lanes)
            offs.append(s)
            ids.append(sid)
        offs = np.asarray(offs)
        ids = np.asarray(ids)
        valid = np.isfinite(offs)
        if valid.sum() < 20:
            continue
        off = offs[valid]
        n_scene += 1
        all_off.append(off)
        n_ids = len(np.unique(ids[valid][ids[valid] >= 0]))
        i0 = int(np.argmax(valid))
        i1 = int(len(valid) - 1 - np.argmax(valid[::-1]))
        delta = float(off[-1] - off[0])
        rec = dict(w=w, max_abs=float(np.abs(off).max()),
                   delta=abs(delta), swing=float(off.max() - off.min()),
                   n_ids=n_ids, at_risk=float(np.abs(off).max()),
                   points=int(valid.sum()))
        per_traj.append(rec)
        if len(examples) < args.plot_n and abs(delta) > 1.0:
            examples.append((w, np.arange(valid.sum()), off))
        if len(examples) < args.plot_n and n_scene % 7 == 0:
            examples.append((w, np.arange(valid.sum()), off))

    if not per_traj:
        raise SystemExit("没有有效场景")

    off = np.concatenate(all_off)
    A = np.abs(off)
    R = per_traj
    mx = np.asarray([r["max_abs"] for r in R])
    dl = np.asarray([r["delta"] for r in R])
    sw = np.asarray([r["swing"] for r in R])
    nid = np.asarray([r["n_ids"] for r in R])

    def pct(a, ps=(50, 75, 90, 95, 99)):
        return "  ".join(f"p{p}={np.percentile(a, p):.2f}" for p in ps)

    print(f"有效场景 {n_scene}，横向偏移采样点 {len(off)} 个")
    print()
    print("【1】相对车道中心的横向偏移 |offset|（m）:")
    print(f"   {pct(A)}")
    print(f"   |offset| > 0.5 m 的点: {100*np.mean(A > 0.5):.1f}%   "
          f">1.0 m: {100*np.mean(A > 1.0):.1f}%   >1.5 m: {100*np.mean(A > 1.5):.1f}%")
    print()
    print("【2】单条轨迹（m）:")
    print(f"   最大|offset|:  {pct(mx)}")
    print(f"   净横向位移|Δ|: {pct(dl)}")
    print(f"   横向摆动幅度:  {pct(sw)}")
    print()
    print("【3】横向行为事件（关键判据）:")
    for th in (0.3, 0.5, 1.0, 1.5, 2.0):
        print(f"   |Δoffset| > {th:.1f} m 的轨迹: {100*np.mean(dl > th):5.1f}%")
    print(f"   最近车道 id 发生切换（疑似换道）: {100*np.mean(nid > 1):5.1f}%")
    print(f"   单条轨迹涉及车道数: p50={np.percentile(nid,50):.0f} "
          f"max={nid.max()}")
    print()
    print("【4】横移速度（|d offset/dt|）:")
    dots = []
    for o in all_off:
        if len(o) > 5:
            dots.append(np.abs(np.diff(o)) / 0.1)
    if dots:
        dots = np.concatenate(dots)
        print(f"   {pct(dots)} m/s")
        print(f"   >0.5 m/s 的点: {100*np.mean(dots > 0.5):.1f}%   "
              f">1.0 m/s: {100*np.mean(dots > 1.0):.1f}%")

    print()
    print("=== 判定 ===")
    if np.percentile(dl, 95) < 0.3 and np.mean(nid > 1) < 0.05:
        print("  数据里几乎没有横向行为 → 横向候选价值有限")
    elif np.percentile(dl, 95) < 1.0:
        print("  横向行为存在但幅度小（p95 < 1 m）→ 横向候选只对'小幅避让'有意义")
    else:
        print("  横向行为明显（p95 ≥ 1 m）→ 值得学")

    if args.out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(args.out, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
        axes[0, 0].hist(off, bins=100, color="#3182bd")
        axes[0, 0].set_xlabel("signed lateral offset to lane center (m)")
        axes[0, 0].set_ylabel("count")
        axes[0, 0].set_title("all sampled points")
        axes[0, 0].grid(alpha=0.25)

        axes[0, 1].hist(dl, bins=60, color="#e6550d")
        axes[0, 1].axvline(1.0, color="#a50f15", ls="--", label="1.0 m")
        axes[0, 1].set_xlabel("|net lateral shift| per trajectory (m)")
        axes[0, 1].set_title("lateral maneuver magnitude")
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].grid(alpha=0.25)

        xs = np.sort(dl)
        axes[1, 0].plot(xs, np.linspace(0, 1, len(xs)), lw=2, color="#31a354")
        axes[1, 0].axvline(1.0, color="#a50f15", ls="--")
        axes[1, 0].set_xlabel("|net lateral shift| (m)")
        axes[1, 0].set_ylabel("CDF")
        axes[1, 0].set_title("CDF of lateral maneuver magnitude")
        axes[1, 0].grid(alpha=0.25)

        for (w, idx, o) in examples[:4]:
            axes[1, 1].plot(idx * 0.1, o, lw=1.6, label=f"world {w}")
        axes[1, 1].axhline(0, color="#999999", lw=0.8)
        axes[1, 1].set_xlabel("time (s)")
        axes[1, 1].set_ylabel("lateral offset (m)")
        axes[1, 1].set_title("examples: offset vs time")
        axes[1, 1].legend(fontsize=7)
        axes[1, 1].grid(alpha=0.25)
        fig.tight_layout()
        fn = os.path.join(args.out, "lateral_offset.png")
        fig.savefig(fn, dpi=120)
        plt.close(fig)
        print("saved", fn)


if __name__ == "__main__":
    main()
