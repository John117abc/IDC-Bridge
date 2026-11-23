"""在 GPUDrive 里跑「上层规划器 → IDC 参考」，采集 IDC v2 状态对。

产物（npz）: state(T,84) / refpos(T,41,2) / ego(T,3) / step / world / replan

用法（容器内）:
  /gpudrive/.venv/bin/python /gpudrive/baselines/IDC-Bridge/tools/collect_v2_pairs.py \
      --ckpt /workspace/data/checkpoints/planner.pt --out /workspace/data/v2_pairs.npz \
      --worlds 8 --steps 200
"""
import argparse
import os
import random
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO, os.path.join(REPO, 'src'), os.path.join(REPO, 'tools')):
    if p not in sys.path:
        sys.path.insert(0, p)

from env.env_utils import get_env_config, load_scenes, get_ego_indices  # noqa: E402
from env.candidate_sampler import install_external  # noqa: E402
from env.planner_runtime import PlannerRuntime  # noqa: E402
from gpudrive.env.env_torch import GPUDriveTorchEnv  # noqa: E402
from gpudrive.env.dataset import SceneDataLoader  # noqa: E402
from env.idc_state_builder import GPUDriveObservationBuilder  # noqa: E402
import idc_state_v2 as V2  # noqa: E402


def partner_obs_to_env(env, w, a, heading, x, y):
    p = env.sim.partner_observations_tensor().to_torch().cpu().numpy()[w, a]
    out = []
    c, s = np.cos(heading), np.sin(heading)
    for row in p:
        v, rx, ry, rh = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        if v == 0.0 and rx == 0.0 and ry == 0.0:
            continue
        out.append(dict(x=x + rx * c - ry * s, y=y + rx * s + ry * c,
                        h=heading + rh, v=v))
    return out


def path_from_env_ref(ref, dt=0.1, n=41):
    pos = np.asarray(ref, np.float32)[:n]
    d = np.diff(pos, axis=0)
    heading = np.r_[np.unwrap(np.arctan2(d[:, 1], d[:, 0])), 0.0]
    heading[-1] = heading[-2]
    speed = np.r_[np.linalg.norm(d, axis=1) / dt, 0.0]
    speed[-1] = speed[-2]
    kappa = np.zeros(len(pos), np.float32)
    for i in range(1, len(pos) - 1):
        a = pos[i] - pos[i - 1]
        b = pos[i + 1] - pos[i]
        cross = a[0] * b[1] - a[1] * b[0]
        den = max(np.linalg.norm(a) * np.linalg.norm(b)
                  * np.linalg.norm(pos[i + 1] - pos[i - 1]), 1e-6)
        kappa[i] = abs(cross) / den
    return dict(pos=pos, heading=heading.astype(np.float32),
                speed=speed.astype(np.float32), kappa=kappa, dt=dt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', default='/workspace/data/v2_pairs.npz')
    ap.add_argument('--worlds', type=int, default=8)
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--k', type=int, default=1)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--max_agents', type=int, default=8)
    ap.add_argument('--data_root', default='')
    args = ap.parse_args()

    from utils.config import build_config
    cfg = build_config(os.path.join(REPO, 'src', 'configs', 'eval.yaml'))
    cfg.num_worlds = args.worlds
    cfg.device = args.device
    cfg.max_agents = args.max_agents
    cfg.epochs = 1
    config = cfg

    data_loader = SceneDataLoader(
        root=args.data_root or getattr(config, 'file_dir', None) or '',
        batch_size=args.worlds,
        dataset_size=args.worlds,
        sample_with_replacement=True,
        shuffle=True,
        seed=42,
    )
    env_config = get_env_config()
    env = GPUDriveTorchEnv(config=env_config, data_loader=data_loader,
                           max_cont_agents=args.max_agents,
                           device=args.device, action_type='continuous')
    scenes = load_scenes(env)
    builder = GPUDriveObservationBuilder(env, scenes)
    ego_indices = get_ego_indices(env, args.worlds)
    builder.generate_candidate_paths(ego_indices, num_paths=1)
    print('[collect] env 就绪, worlds =', args.worlds, 'file_dir =', getattr(config, 'file_dir', ''), flush=True)

    RT = PlannerRuntime(args.ckpt, k=args.k, replan_every=5,
                        device=args.device, mode='planner')
    RT.debug = True

    def install(b, ego_idx, paths, speeds, tgt):
        maps = env.sim.map_observation_tensor().to_torch().cpu().numpy()
        install_external(b, ego_idx, paths, speeds, target_pos=tgt,
                         last_valid_idx={w: 41 for w in paths},
                         blend_tail=False, maps=maps)

    S, RP, EG, ST, WD, RE = [], [], [], [], [], []
    env.reset()
    RT.reset()
    for step in range(args.steps):
        RT.observe(env, ego_indices)
        did = False
        if step % max(RT.replan_every, 1) == 0:
            RT.maybe_replan(step, env, builder, ego_indices, install)
            did = True
        ab = env.sim.absolute_self_observation_tensor().to_torch().cpu().numpy()
        rl = env.sim.self_observation_tensor().to_torch().cpu().numpy()
        for w in range(args.worlds):
            a = int(ego_indices[w])
            x, y, h = float(ab[w, a, 0]), float(ab[w, a, 1]), float(ab[w, a, 7])
            v = float(rl[w, a, 0])
            ref = RT.last.get(w)
            if ref is None:
                try:
                    ref = builder.candidate_paths[w][a][0]['pos']
                except Exception:
                    continue
            path = path_from_env_ref(ref)
            nbrs = partner_obs_to_env(env, w, a, h, x, y)
            st = V2.build_state_v2(
                dict(v=v, v_prev=v, omega=0.0, a_lat=0.0, steer_prev=0.0,
                     acc_prev=0.0, pose=(x, y, h)),
                nbrs, path, replan_phase=(step % RT.replan_every) / 5.0)
            S.append(st)
            RP.append(path['pos'])
            EG.append([x, y, h])
            ST.append(step)
            WD.append(w)
            RE.append(1.0 if did else 0.0)
        act = np.zeros((args.worlds, args.max_agents, 2), np.float32)
        env.step(act)
        if step % 20 == 0:
            print(f'[collect] step {step} 样本 {len(S)}', flush=True)

    S = np.stack(S)
    np.savez_compressed(args.out, state=S, refpos=np.stack(RP),
                        ego=np.stack(EG), step=np.array(ST),
                        world=np.array(WD), replan=np.array(RE))
    print(f'[collect] 保存 {S.shape} -> {args.out}', flush=True)


if __name__ == '__main__':
    main()
