"""诊断参考方向是否与专家相反：逐个 world 打印位姿、专家位移、候选位移。

用法（容器内）：
    python tools/check_ref_direction.py --data-dir /data/data_json/training \
        --num-worlds 16 --seed 0
"""
import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import numpy as np  # noqa: E402

OFFSETS = (-3.5, -1.6, -0.8, 0.0, 0.8, 1.6, 3.5)


def ang(v):
    return float(np.degrees(np.arctan2(v[1], v[0])))


def wrap(a):
    return float((np.degrees(a) + 180) % 360 - 180)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--config", default=os.path.join(_ROOT, "src", "configs", "eval.yaml"))
    ap.add_argument("--num-worlds", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
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
    maps = env.sim.map_observation_tensor().to_torch().cpu().numpy()
    expert = builder.expert_pos.detach().cpu().numpy()
    abs_np = env.sim.absolute_self_observation_tensor().to_torch().cpu().numpy()
    scfg = SamplerConfig()

    print(f"{'w':>3s} {'ego(x,y)':>16s} {'hdr7':>7s} {'hdr专家':>8s} "
          f"{'专家位移角':>10s} {'候选末位移角':>12s} {'夹角':>7s} {'专家长':>7s} {'候选长':>7s}")
    for w in range(cfg.num_worlds):
        a = int(ego[w])
        ex = expert[w, a]
        ok = np.all(np.abs(ex) < 1e4, axis=1)
        exv = ex[ok]
        x, y = float(abs_np[w, a, 0]), float(abs_np[w, a, 1])
        hdr7 = float(abs_np[w, a, 7])
        if len(exv) < 3:
            print(f"{w:3d} 专家轨迹有效点不足 ({len(exv)})")
            continue
        d_ex = exv[-1] - exv[0]
        L_ex = np.linalg.norm(d_ex)
        hdr_ex = np.degrees(np.arctan2(exv[1, 1] - exv[0, 1], exv[1, 0] - exv[0, 0]))
        cands, base = generate_map_candidates((x, y, hdr7), maps[w], (0.0,),
                                              scfg, speed0=8.0)
        d_c = base[-1] - base[0]
        L_c = np.linalg.norm(d_c)
        ang_c = ang(d_c)
        ang_e = ang(d_ex)
        diff = wrap(np.radians(ang_c - ang_e))
        flag = "  <<< 反向" if abs(diff) > 90 else ""
        print(f"{w:3d} ({x:7.1f},{y:6.1f}) {np.degrees(hdr7):7.1f} "
              f"{hdr_ex:8.1f} {ang_e:10.1f} {ang_c:12.1f} {diff:7.1f} "
              f"{L_ex:7.1f} {L_c:7.1f}{flag}")


if __name__ == "__main__":
    main()
