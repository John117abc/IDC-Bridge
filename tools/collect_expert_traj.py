"""采集专家轨迹并转换到自车系（起点原点、航向 0），供离线 anchor 聚类。

一个进程只能创建一个 GPUDriveTorchEnv（madrona 引擎是进程级单例），
所以按 seed 分进程采集，每次一批。输出 npz 到 --out-dir。

用法（容器内）：
    python tools/collect_expert_traj.py --data-dir /data/data_json/training \
        --seed 0 --num-worlds 120 --out-dir /workspace/data/anchors
"""
import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import numpy as np  # noqa: E402

T = 91
DT = 0.1


def to_ego_frame(p, h0):
    """世界坐标 -> 自车系（起点为原点，航向 0；x 向前，y 向左）。"""
    d = p - p[0]
    c, s = np.cos(-h0), np.sin(-h0)
    x = d[:, 0] * c - d[:, 1] * s
    y = d[:, 0] * s + d[:, 1] * c
    return np.stack([x, y], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-worlds", type=int, default=120)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    from gpudrive.env.dataset import SceneDataLoader
    from gpudrive.env.env_torch import GPUDriveTorchEnv

    from env.env_utils import get_ego_indices, get_env_config

    loader = SceneDataLoader(root=args.data_dir, batch_size=args.num_worlds,
                             dataset_size=100000,
                             sample_with_replacement=False, shuffle=True,
                             seed=args.seed)
    env = GPUDriveTorchEnv(config=get_env_config(), data_loader=loader,
                           max_cont_agents=1, device="cuda",
                           action_type="continuous")
    env.reset()
    ego = get_ego_indices(env, args.num_worlds)

    raw = env.sim.expert_trajectory_tensor().to_torch().cpu().numpy()
    W, A, _ = raw.shape
    pos = raw[:, :, :2 * T].reshape(W, A, T, 2)

    trajs, lens, speeds, worlds = [], [], [], []
    for w in range(W):
        p = pos[w, ego[w]]
        ok = (np.abs(p[:, 0]) < 1e3) & (np.abs(p[:, 1]) < 1e3)
        if not ok.all() or not ok.any():
            # 只保留整条有效的轨迹（聚类需要同一时域）
            continue
        d = np.diff(p, axis=0)
        h0 = float(np.arctan2(d[0, 1], d[0, 0]))
        trajs.append(to_ego_frame(p, h0).astype(np.float32))
        speeds.append((np.linalg.norm(d, axis=1) / DT).astype(np.float32))
        lens.append(float(np.sum(np.linalg.norm(d, axis=1))))
        worlds.append(w)

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"traj_seed{args.seed:03d}.npz")
    np.savez_compressed(
        out,
        traj=np.asarray(trajs),               # [N, T, 2] 自车系
        speed=np.asarray(speeds),             # [N, T-1]
        length=np.asarray(lens),              # [N] 总里程
        world=np.asarray(worlds),
        seed=np.asarray(args.seed),
    )
    print(f"[collect] seed={args.seed} worlds={W} kept={len(trajs)} -> {out}")


if __name__ == "__main__":
    main()
