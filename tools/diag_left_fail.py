"""诊断：为什么 Left +3.5 跟不住，而 Right -3.5 跟得很好？

对每条候选打印：
  • 几何：弧长、末端 road_dist（到边界距离）
  • 速度剖面：均值 / 最小值（看是不是曲率限速把速度压掉了）
  • 车辆：实际行驶距离、末段 |lat|
  • 边界假设 vs 曲率假设的判别：Left+3.5 的 road_dist 是否比 Right-3.5 小

用法（容器内）：
    python tools/diag_left_fail.py --data-dir /data/data_json/training \
        --model-path <pth> --num-worlds 8 --seed 0
"""
import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import numpy as np  # noqa: E402

OFFSETS = (-3.5, -1.6, -0.8, 0.0, 0.8, 1.6, 3.5)


def label_of(o):
    if abs(o) < 1e-6:
        return "Center"
    side = "Left " if o > 0 else "Right"
    return f"{side} {o:+.1f}"


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
    K = install_fixed(builder, ego, maps, OFFSETS)

    print("=" * 96)
    print("PART A: 每条候选的几何 / 速度 / 边界距离（对所有 world 取中位）")
    print("=" * 96)
    hdr = f"{'candidate':12s} {'弧长':>7s} {'road_end':>9s} {'v_mean':>7s} {'v_min':>6s} {'κ_max':>7s}"
    print(hdr)
    for k in range(K):
        arcs, rds, vm, vmin, kmax = [], [], [], [], []
        for w in range(cfg.num_worlds):
            a = int(ego[w])
            p = builder.candidate_paths[w][a][k]
            pos = np.asarray(p['pos'], dtype=np.float64)
            spd = np.asarray(p['speed'], dtype=np.float64)
            rd = np.asarray(p['road_dist'], dtype=np.float64)
            arcs.append(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum())
            rds.append(np.median(rd[-10:]))
            vm.append(spd.mean())
            vmin.append(spd.min())
            d = np.diff(pos, axis=0)
            hd = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
            ds = np.maximum(np.linalg.norm(d, axis=1), 1e-6)
            kap = np.abs(np.diff(hd) / (0.5 * (ds[1:] + ds[:-1])))
            kmax.append(np.percentile(kap, 95) if len(kap) else 0.0)
        print(f"{label_of(OFFSETS[k]):12s} {np.median(arcs):7.1f} "
              f"{np.median(rds):9.2f} {np.median(vm):7.2f} "
              f"{np.median(vmin):6.2f} {np.median(kmax):7.4f}")

    print()
    print("=" * 96)
    print("PART B: 每个 world 的 lat 时间序列（step 10/30/50/70/90），Left+3.5 vs Right-3.5")
    print("=" * 96)
    agent = DiscreteIDCAgent(env, cfg, cfg.device, builder, ego)
    agent.load(cfg.model_path)
    ref_start = agent.DIM_EGO + agent.DIM_OTHERS + agent.DIM_VALIDITY
    IX_LAT = ref_start + 3
    T = builder.EXPERT_TRAJ_LEN
    probes = [10, 30, 50, 70, 90]

    for k in (0, 6):
        env.reset()
        series = {w: {} for w in range(cfg.num_worlds)}
        dist = np.zeros(cfg.num_worlds)
        prev = None
        for step in range(T):
            states = builder.get_idc_observations_batch(
                ego, path_indices=[k] * cfg.num_worlds)
            pos = np.array([[float(s[0]), float(s[1])] for s in states])
            if prev is not None:
                dist += np.linalg.norm(pos - prev, axis=1)
            prev = pos
            if step in probes:
                for w in range(cfg.num_worlds):
                    series[w][step] = float(states[w][IX_LAT])
            acts = agent.select_action(states, deterministic=True)
            env.step_dynamics(extend_action_to_3d(acts))
        print(f"\n--- {label_of(OFFSETS[k])} ---")
        print(f"{'world':>6s} " + " ".join(f"{'s'+str(s):>8s}" for s in probes)
              + f" {'行驶距离':>9s}")
        for w in range(cfg.num_worlds):
            vals = " ".join(f"{series[w].get(s, float('nan')):8.2f}" for s in probes)
            print(f"{w:>6d} {vals} {dist[w]:9.1f}")


if __name__ == "__main__":
    main()
