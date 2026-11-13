"""判定：本镜像里 sim.set_maps (swap_data_batch) 到底能不能用。"""
import sys

import torch

sys.path.insert(0, '/gpudrive/baselines/IDC-Bridge/src')
from gpudrive.env.env_torch import GPUDriveTorchEnv           # noqa: E402
from gpudrive.env.dataset import SceneDataLoader              # noqa: E402
from env.env_utils import get_env_config                      # noqa: E402


def names(batch):
    return [p.split('/')[-1] for p in batch[:3]]


loader = SceneDataLoader(root='/data/data_json/training', batch_size=8,
                         dataset_size=400, sample_with_replacement=False,
                         shuffle=True, seed=0)
env = GPUDriveTorchEnv(config=get_env_config(), data_loader=loader,
                       max_cont_agents=1, device='cuda',
                       action_type='continuous')
print('batch#1:', names(env.data_batch), flush=True)

env.reset()
print('reset #1 OK', flush=True)

# 走一步，模拟"训练过一会儿再换场景"
for _ in range(5):
    a = torch.zeros(8, 1, 3, device='cuda')
    a[..., 0] = 0.5
    env.step_dynamics(a)
print('5 steps OK', flush=True)

env.swap_data_batch()
print('swap_data_batch OK  batch#2:', names(env.data_batch), flush=True)

env.reset()
print('reset #2 OK', flush=True)

for _ in range(3):
    a = torch.zeros(8, 1, 3, device='cuda')
    env.step_dynamics(a)
print('post-swap steps OK', flush=True)
print('ALL GOOD')
