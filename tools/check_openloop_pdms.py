"""验证开环 PDMS 评测器。

检查三件事：
  1. 人类专家轨迹自评 → 应该接近满分（这是参照系）
  2. 给专家轨迹加横向噪声 σ=0.3/0.6/1.2 m → PDMS 应单调下降
  3. 给专家轨迹加纵向缩放 0.8/1.2 → PDMS 应下降（进度/碰撞受影响）

用法（容器内，CPU 即可）：
    python tools/check_openloop_pdms.py --data-dir /data/data_json/training \
        --n-scenes 100 --seed 0
"""
import argparse
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def smoothstep5(u):
    u = np.clip(u, 0, 1)
    return u ** 3 * (10 - 15 * u + 6 * u * u)


def lateral_noise(traj, sigma, rng):
    d = np.diff(traj, axis=0)
    t = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    t = np.vstack([t, t[-1:]])
    n = np.stack([-t[:, 1], t[:, 0]], 1)
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(d, axis=1))]
    u = arc / max(arc[-1], 1e-6)
    scale = rng.uniform(0.5, 1.0) * smoothstep5(u)
    return traj + (sigma * scale)[:, None] * n


def longitudinally_scale(traj, s):
    """沿自身弧长重采样到 s 倍长度（模拟走太快/太慢）。"""
    d = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    arc = np.r_[0.0, np.cumsum(d)]
    total = max(arc[-1], 1e-6)
    target = np.linspace(0, total * s, len(traj))
    x = np.interp(target, arc, traj[:, 0])
    y = np.interp(target, arc, traj[:, 1])
    return np.stack([x, y], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--n-scenes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from metrics.openloop_pdms import SceneCache, score_trajectory

    print("加载场景（纯 CPU，读 JSON）...")
    cache = SceneCache(args.data_dir, n_scenes=args.n_scenes, seed=args.seed)
    print(f"有效场景: {len(cache.scenes)}")
    rng = np.random.default_rng(args.seed)

    cases = {
        '专家轨迹(自评)': lambda tr: tr,
        '+横向噪声 0.3': lambda tr: lateral_noise(tr, 0.3, rng),
        '+横向噪声 0.6': lambda tr: lateral_noise(tr, 0.6, rng),
        '+横向噪声 1.2': lambda tr: lateral_noise(tr, 1.2, rng),
        '+纵向 x0.8': lambda tr: longitudinally_scale(tr, 0.8),
        '+纵向 x1.2': lambda tr: longitudinally_scale(tr, 1.2),
    }

    print()
    print(f"{'case':18s} {'PDMS':>7s} {'NC':>5s} {'DAC':>5s} {'EP':>6s} "
          f"{'TTC':>6s} {'Cmft':>6s} {'LK':>6s} {'碰撞步':>6s}")
    for name, fn in cases.items():
        rng = np.random.default_rng(args.seed)
        scores, comps = [], {k: [] for k in ('nc', 'dac', 'ep', 'ttc',
                                             'comfort', 'lk', 'collision_steps')}
        for sc in cache.scenes:
            ex = sc.expert_future      # 未来窗口（帧 11..90）
            ok = np.all(np.abs(ex) < 1e4, axis=1)
            if ok.sum() < 20:
                continue
            ex = ex[:int(np.argmin(ok)) if not ok.all() else len(ex)]
            if len(ex) < 20:
                continue
            tr = fn(ex.copy())
            r = score_trajectory(sc, tr)
            if not r.get('valid'):
                continue
            scores.append(r['driving_score'])
            for k in comps:
                comps[k].append(r[k])
        if not scores:
            print(f"{name:18s}  无有效样本")
            continue
        print(f"{name:18s} {np.mean(scores):7.1f} {np.mean(comps['nc']):5.2f} "
              f"{np.mean(comps['dac']):5.2f} {np.mean(comps['ep']):6.3f} "
              f"{np.mean(comps['ttc']):6.3f} {np.mean(comps['comfort']):6.3f} "
              f"{np.mean(comps['lk']):6.3f} {np.mean(comps['collision_steps']):6.2f}")


if __name__ == "__main__":
    main()
