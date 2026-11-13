"""判定：同一进程里能否构造第二个 GPUDriveTorchEnv（用于"每 N 轮换一批场景"）。"""
import sys

import torch

sys.path.insert(0, '/gpudrive/baselines/IDC-Bridge/src')
from gpudrive.env.env_torch import GPUDriveTorchEnv           # noqa: E402
from gpudrive.env.dataset import SceneDataLoader              # noqa: E402
from env.env_utils import get_env_config                      # noqa: E402


def make_env(seed, n=8):
    loader = SceneDataLoader(root='/data/data_json/training', batch_size=n,
                             dataset_size=400, sample_with_replacement=False,
                             shuffle=True, seed=seed)
    return GPUDriveTorchEnv(config=get_env_config(), data_loader=loader,
                            max_cont_agents=1, device='cuda',
                            action_type='continuous')


env1 = make_env(0)
env1.reset()
a = torch.zeros(8, 1, 3, device='cuda')
for _ in range(3):
    env1.step_dynamics(a)
print('env#1 OK, first file:', env1.data_batch[0].split('/')[-1], flush=True)

# 尝试在同一进程里再建一个引擎
env2 = make_env(1)
env2.reset()
for _ in range(3):
    env2.step_dynamics(a)
print('env#2 OK, first file:', env2.data_batch[0].split('/')[-1], flush=True)

# 回到 env1 继续用（确认第一个引擎没被破坏）
env1.reset()
for _ in range(3):
    env1.step_dynamics(a)
print('env#1 reuse OK', flush=True)
print('SECOND_ENV_WORKS')
