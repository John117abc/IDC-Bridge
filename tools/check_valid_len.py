"""§6.2 验收 0b：只改无效尾段时，状态 / critic 分数 / 首动作应不变。"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, '/gpudrive/baselines/IDC-Bridge/src')
from env.candidate_sampler import install_external  # noqa


def build_ref(px, py, ph, speed, n_valid=40, total=91, turn_tail=False):
    """前 n_valid+1 个点固定（直线），之后一份直行、一份急转。"""
    pts = []
    th = ph
    x, y = px, py
    for i in range(total):
        pts.append([x, y])
        if i >= n_valid and turn_tail:
            th += 0.35            # 急转尾段
        x += speed * 0.1 * np.cos(th)
        y += speed * 0.1 * np.sin(th)
    P = np.asarray(pts, np.float32)[None]     # (1, N, 2)
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--data-dir', default='/data/data_json/training')
    ap.add_argument('--num-worlds', type=int, default=3)
    args = ap.parse_args()

    import torch
    from gpudrive.env.config import EnvConfig
    from gpudrive.env.env_torch import GPUDriveTorchEnv
    from gpudrive.datatypes.observation import GlobalObservations
    from gpudrive.visualize.utils import img_from_device, deltas_to_world
    from agents.idc_agent import DiscreteIDCAgent
    from env.idc_state_builder import IDCStateBuilder
    from utils.config import build_config
    from gpudrive.datatypes.scene import SceneDataLoader

    cfg = build_config('/gpudrive/baselines/IDC-Bridge/src/configs/pcont_tr.yaml')
    cfg.num_worlds = args.num_worlds
    cfg.device = 'cuda:0'

    env_config = EnvConfig()
    dataset = SceneDataLoader(root=args.data_dir, batch_size=args.num_worlds,
                              dataset_size=args.num_worlds,
                              sample_with_replacement=False, shuffle=False,
                              seed=42)
    env = GPUDriveTorchEnv(config=env_config, data_loader=dataset,
                           max_cont_agents=cfg.max_agents, device=cfg.device)
    env.reset()

    builder = IDCStateBuilder(env.sim if hasattr(env, 'sim') else env,
                              num_worlds=args.num_worlds,
                              num_candidates=1, num_points=91)
    ego = env.sim.absolute_self_observation_tensor().to_torch().cpu().numpy()
    maps = env.sim.map_observation_tensor().to_torch().cpu().numpy()

    paths, speeds = {}, {}
    for w in range(args.num_worlds):
        px, py, ph = float(ego[w, 0, 0]), float(ego[w, 0, 1]), float(ego[w, 0, 7])
        paths[w] = build_ref(px, py, ph, 10.0, turn_tail=False)
        speeds[w] = np.full((1, 91), 10.0, np.float32)

    def make_states(turn):
        pp = {}
        for w in range(args.num_worlds):
            px, py, ph = float(ego[w, 0, 0]), float(ego[w, 0, 1]), float(ego[w, 0, 7])
            pp[w] = build_ref(px, py, ph, 10.0, turn_tail=turn)
        install_external(builder, list(range(args.num_worlds)), pp,
                         speeds_by_world=speeds, maps=maps,
                         last_valid_idx={w: 41 for w in range(args.num_worlds)})
        st = builder.get_idc_observations_batch(
            list(range(args.num_worlds)), path_indices=[0] * args.num_worlds)
        return np.stack([st[w] for w in range(args.num_worlds)])

    s_a = make_states(False)
    s_b = make_states(True)
    d_state = float(np.max(np.abs(s_a - s_b)))

    agent = DiscreteIDCAgent(env, cfg, cfg.device, builder,
                             list(range(args.num_worlds)))
    agent.load(args.model)
    with torch.no_grad():
        qa = agent._value(torch.from_numpy(s_a).float().to(cfg.device)).cpu().numpy()
        qb = agent._value(torch.from_numpy(s_b).float().to(cfg.device)).cpu().numpy()
    d_q = float(np.max(np.abs(qa - qb)))

    print(f'[0b] state max|diff| = {d_state:.3e}')
    print(f'[0b] critic max|diff| = {d_q:.3e}')
    print('[0b] PASS' if d_state < 1e-5 and d_q < 1e-4 else '[0b] FAIL')


if __name__ == '__main__':
    main()
