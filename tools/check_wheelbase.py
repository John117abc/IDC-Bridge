"""D2 验证：GPUDrive 的 classic 动力学实际把什么当成轴距。

GPUDrive 的 forwardKinematics 里横摆增量是
    dpsi = v * cos(beta) * tan(delta) / size.length * dt,   beta = atan(0.5*tan(delta))
所以给定 v、delta、dt，测出 dpsi 就能反解出它实际用的 size.length。
这比读字段更可靠——测的是动力学真正在用的那个数。

用法（容器内）：
    python tools/check_wheelbase.py --data-dir /data/data_json/training
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from gpudrive.env.env_torch import GPUDriveTorchEnv          # noqa: E402
from gpudrive.env.dataset import SceneDataLoader             # noqa: E402
from env.env_utils import get_env_config, load_scenes, get_ego_indices  # noqa: E402

DT = 0.1          # forwardKinematics 里硬编码的 dt
AGENT_SCALE = 0.7  # consts::vehicleLengthScale


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--num-worlds', type=int, default=8)
    args = ap.parse_args()

    loader = SceneDataLoader(root=args.data_dir, batch_size=args.num_worlds,
                             dataset_size=32, sample_with_replacement=False,
                             shuffle=True, seed=5)
    env = GPUDriveTorchEnv(config=get_env_config(), data_loader=loader,
                           max_cont_agents=1, device='cuda',
                           action_type='continuous')
    env.reset()
    scenes = load_scenes(env)
    ego = get_ego_indices(env, args.num_worlds)
    W = args.num_worlds

    # ---------- 来源 1: 数据集 JSON 里的 length ----------
    print('=== 来源1: 场景 JSON 里 ego 的 length 字段 ===')
    json_len = []
    for w in range(W):
        a = ego[w]
        obj = scenes[w]['objects'][a]
        ln = obj.get('length')
        json_len.append(ln)
        print(f'  world_{w} agent_{a}  length={ln}  width={obj.get("width")}')
    print(f'  中位数 = {np.median(json_len):.3f}')

    # ---------- 来源 2: 观测张量的第 10 维 ----------
    abs_obs = env.sim.absolute_self_observation_tensor().to_torch().cpu().numpy()
    obs_len = np.array([abs_obs[w, ego[w], 10] for w in range(W)], dtype=np.float64)
    print('\n=== 来源2: abs_obs[..., 10]（observation.py 当作 vehicle_length） ===')
    for w in range(W):
        print(f'  world_{w}: obs[10]={obs_len[w]:.3f}   '
              f'obs[10]*0.7={obs_len[w] * AGENT_SCALE:.3f}')
    print(f'  中位数 obs[10]={np.median(obs_len):.3f}, '
          f'obs[10]*0.7={np.median(obs_len) * AGENT_SCALE:.3f}')

    # ---------- 来源 3: 行为测量（权威） ----------
    # 先给几脚油把速度建立起来（部分场景首帧速度为 0）
    for _ in range(6):
        a3 = torch.zeros(W, 1, 3, device='cuda')
        a3[..., 0] = 1.2
        env.step_dynamics(a3)

    def read_state():
        ab = env.sim.absolute_self_observation_tensor().to_torch().cpu().numpy()
        se = env.sim.self_observation_tensor().to_torch().cpu().numpy()
        yaw = np.array([ab[w, ego[w], 7] for w in range(W)], dtype=np.float64)
        spd = np.array([se[w, ego[w], 0] for w in range(W)], dtype=np.float64)
        return yaw, spd

    print('\n=== 来源3: 行为反解 L_eff = v*cos(beta)*tan(delta)*dt / dpsi ===')
    print(f"{'delta':>7s} {'world':>6s} {'v(m/s)':>8s} {'dpsi(rad)':>11s} {'L_eff(m)':>9s}")
    est = []
    for delta in (0.10, 0.20, 0.30, -0.20):
        for w in range(W):
            yaw0, spd0 = read_state()
            act = torch.zeros(W, 1, 3, device='cuda')
            act[..., 0] = 0.0          # a=0 -> speed 不变，只剩转向
            act[..., 1] = delta
            env.step_dynamics(act)
            yaw1, spd1 = read_state()
            v = float(spd0[w])
            dpsi = float(wrap(yaw1[w] - yaw0[w]))
            if v < 2.0 or abs(dpsi) < 1e-6:
                continue
            beta = np.arctan(0.5 * np.tan(delta))
            L = v * np.cos(beta) * np.tan(delta) * DT / dpsi
            est.append(L)
            print(f'{delta:7.2f} {w:6d} {v:8.3f} {dpsi:11.5f} {L:9.3f}')

    if est:
        est = np.array(est)
        print(f'\n  L_eff 中位数 = {np.median(est):.3f} m   '
              f'均值 = {est.mean():.3f}   std = {est.std():.3f}   n = {len(est)}')
        print(f'\n=== 对照 ===')
        print(f'  configs/base.yaml 的 wheelbase          = 5.000')
        print(f'  get_expert_steer_batch 里硬编码的 L      = 5.000')
        print(f'  行为反解 L_eff（这份代码真实在用的）      = {np.median(est):.3f}')
        print(f'  数据集 JSON length 中位数                = {np.median(json_len):.3f}')
        print(f'  obs[10] 中位数                           = {np.median(obs_len):.3f}')
        print(f'  obs[10]*0.7 中位数                       = {np.median(obs_len) * AGENT_SCALE:.3f}')
        err = abs(np.median(est) - 5.0) / np.median(est)
        print(f'\n  与 5.0 的相对偏差 = {err * 100:.1f}%  -> '
              f'{"一致，可以保留 5.0" if err < 0.05 else "不一致，必须改成实测值"}')


if __name__ == '__main__':
    main()
