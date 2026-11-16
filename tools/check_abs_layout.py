"""确认 absolute_self_observation_tensor 的字段布局，特别是航向在第几维。"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import numpy as np  # noqa: E402


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--config", default=os.path.join(_ROOT, "src", "configs", "eval.yaml"))
    ap.add_argument("--num-worlds", type=int, default=4)
    args = ap.parse_args()

    from gpudrive.env.dataset import SceneDataLoader
    from gpudrive.env.env_torch import GPUDriveTorchEnv

    from env.env_utils import get_ego_indices, get_env_config, load_scenes
    from env.idc_state_builder import GPUDriveObservationBuilder
    from utils.config import build_config

    cfg = build_config(args.config, cli_overrides={
        "num_worlds": args.num_worlds, "seed": 0, "device": "cuda",
        "gif_enabled": False})
    loader = SceneDataLoader(root=args.data_dir, batch_size=cfg.num_worlds,
                             dataset_size=cfg.num_worlds,
                             sample_with_replacement=False, shuffle=True,
                             seed=0)
    env = GPUDriveTorchEnv(config=get_env_config(), data_loader=loader,
                           max_cont_agents=1, device=cfg.device,
                           action_type="continuous")
    scenes = load_scenes(env)
    builder = GPUDriveObservationBuilder(env, scenes)
    ego = get_ego_indices(env, cfg.num_worlds)
    expert = builder.expert_pos.detach().cpu().numpy()
    abs_np = env.sim.absolute_self_observation_tensor().to_torch().cpu().numpy()

    print("abs shape:", abs_np.shape)
    for w in range(cfg.num_worlds):
        a = int(ego[w])
        row = abs_np[w, a]
        ex = expert[w, a]
        d = ex[1] - ex[0]
        h_true = float(np.arctan2(d[1], d[0]))
        print(f"\nworld {w}: abs[:10] = " + " ".join(f"{v:+.3f}" for v in row[:10]))
        print(f"   expert 首点=({ex[0,0]:+.2f},{ex[0,1]:+.2f}) "
              f"次点=({ex[1,0]:+.2f},{ex[1,1]:+.2f}) → 真实航向={h_true:+.3f} rad")
        for i in range(min(10, row.shape[0])):
            if abs(np.arctan2(np.sin(row[i] - h_true), np.cos(row[i] - h_true))) < 0.05:
                print(f"   >>> 第 {i} 维与真实航向一致")


if __name__ == "__main__":
    main()
