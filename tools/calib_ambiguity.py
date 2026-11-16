"""标定"起步歧义"阈值：地图侧指标 vs 专家侧真实转角。

对每个 world 打印：
  n_cross : ego 起点 10 m 内、与朝向夹角 40~140° 的车道段数量（地图侧）
  min_ang : 这些段里最小的夹角
  专家转角: 专家净位移方向 与 ego 初始朝向 的夹角（专家侧真值，仅用于标定）

用法（容器内）：
    python tools/calib_ambiguity.py --data-dir /data/data_json/training \
        --num-worlds 32 --seed 0
"""
import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--config", default=os.path.join(_ROOT, "src", "configs", "eval.yaml"))
    ap.add_argument("--num-worlds", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from gpudrive.env.dataset import SceneDataLoader
    from gpudrive.env.env_torch import GPUDriveTorchEnv

    from env.env_utils import get_ego_indices, get_env_config, load_scenes
    from env.idc_state_builder import GPUDriveObservationBuilder
    from env.map_candidates import route_ambiguity
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

    rows = []
    print(f"{'w':>3s} {'n_cross':>7s} {'min_ang':>7s} {'专家转角':>8s} {'专家长':>7s}")
    for w in range(cfg.num_worlds):
        a = int(ego[w])
        ex = expert[w, a]
        ex = ex[np.all(np.abs(ex) < 1e4, axis=1)]
        if len(ex) < 5:
            continue
        pose = (float(abs_np[w, a, 0]), float(abs_np[w, a, 1]),
                float(abs_np[w, a, 7]))
        n_cross, min_ang = route_ambiguity(pose, maps[w])
        d = ex[-1] - ex[0]
        L = float(np.linalg.norm(d))
        if L < 3.0:                       # 近静止，转角无意义
            continue
        net = float(np.degrees(np.arctan2(d[1], d[0])))
        turn = abs((net - np.degrees(pose[2]) + 180) % 360 - 180)
        rows.append((n_cross, min_ang, turn, L))
        print(f"{w:3d} {n_cross:7d} {min_ang:7.1f} {turn:8.1f} {L:7.1f}")

    if not rows:
        print("无有效样本")
        return
    R = np.asarray(rows)
    print()
    print("=== 候选阈值（过滤掉'专家转角>60°'的歧义场景）===")
    bad = R[:, 2] > 60.0
    print(f"  专家转角 >60° 的场景: {int(bad.sum())}/{len(R)}")
    for thr in (1, 2, 3, 5, 8, 12):
        pred = R[:, 0] >= thr
        tp = int((pred & bad).sum())
        fp = int((pred & ~bad).sum())
        fn = int((~pred & bad).sum())
        print(f"  n_cross>={thr:2d}: 命中={tp:2d} 误杀={fp:3d} 漏掉={fn:2d}")


if __name__ == "__main__":
    main()
