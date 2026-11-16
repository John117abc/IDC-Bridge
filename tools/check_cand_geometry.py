"""诊断：三条候选的末端横向间距是否真的是 1.6 m？以及 +0.8 那条车是否提前终止？

用法（容器内）：
    python tools/check_cand_geometry.py --data-dir /data/data_json/training \
        --model-path <pth> --num-worlds 8 --seed 0
"""
import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import numpy as np  # noqa: E402


def ego_frame(P, p0, h0):
    d = P - p0
    c, s = np.cos(-h0), np.sin(-h0)
    return np.stack([d[:, 0] * c - d[:, 1] * s, d[:, 0] * s + d[:, 1] * c], 1)


def arc_len(P):
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--config", default=os.path.join(_ROOT, "src", "configs", "eval.yaml"))
    ap.add_argument("--num-worlds", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

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

    print("=" * 78)
    print("PART A: 候选几何审计（自车系：x=纵向, y=横向）")
    print("=" * 78)
    expert = builder.expert_pos.detach().cpu().numpy()
    for w in range(cfg.num_worlds):
        a = int(ego[w])
        p0 = expert[w, a, 0]
        d0 = expert[w, a, 1] - expert[w, a, 0]
        h0 = float(np.arctan2(d0[1], d0[0]))
        pe = ego_frame(expert[w, a], p0, h0)
        print(f"\nworld {w}: 专家路径 末端=({pe[-1,0]:7.1f}, {pe[-1,1]:+6.2f}) "
              f"弧长={arc_len(expert[w, a]):6.1f} m")
        for k in range(K):
            P = builder.candidate_paths[w][a][k]['pos'].astype(np.float64)
            pek = ego_frame(P, p0, h0)
            print(f"   cand{k} (期望 y_end={(-0.8,0.0,0.8)[k]:+.1f}): "
                  f"末端=({pek[-1,0]:7.1f}, {pek[-1,1]:+6.2f}) "
                  f"弧长={arc_len(P):6.1f} m  "
                  f"起点误差={np.linalg.norm(P[0]-expert[w,a,0]):.4f} m")

    print()
    print("=" * 78)
    print("PART B: 强制 idx=2 (+0.8) 时，ego 是否提前终止/失效")
    print("=" * 78)
    agent = DiscreteIDCAgent(env, cfg, cfg.device, builder, ego)
    agent.load(cfg.model_path)
    T = builder.EXPERT_TRAJ_LEN

    env.reset()
    bad_first = {}
    for step in range(T):
        states = builder.get_idc_observations_batch(
            ego, path_indices=[2] * cfg.num_worlds)
        if step % 10 == 0 or step == T - 1:
            for w in range(cfg.num_worlds):
                x, y = float(states[w][0]), float(states[w][1])
                if abs(x) > 1e3 or abs(y) > 1e3:
                    bad_first.setdefault(w, step)
            xs = [float(s[0]) for s in states]
            nb = sum(1 for v in xs if abs(v) > 1e3)
            print(f"step {step:3d}: 失效世界数={nb}/{cfg.num_worlds}")
        acts = agent.select_action(states, deterministic=True)
        env.step_dynamics(extend_action_to_3d(acts))

    print()
    print("各世界首次出现哨兵值(|pos|>1e3)的步数:")
    for w in range(cfg.num_worlds):
        print(f"   world {w}: {bad_first.get(w, '未出现')}")


if __name__ == "__main__":
    main()
