"""判定 set_maps 的崩溃是否与"新批内容"有关。

三组：
  A) swap 到完全相同的批（内容不变）
  B) swap 到 8 个同一小场景
  C) 只在 A 成功时才继续，说明是数据相关
"""
import sys

import torch

sys.path.insert(0, '/gpudrive/baselines/IDC-Bridge/src')
from gpudrive.env.env_torch import GPUDriveTorchEnv           # noqa: E402
from gpudrive.env.dataset import SceneDataLoader              # noqa: E402
from env.env_utils import get_env_config                      # noqa: E402

loader = SceneDataLoader(root='/data/data_json/training', batch_size=8,
                         dataset_size=400, sample_with_replacement=False,
                         shuffle=True, seed=0)
env = GPUDriveTorchEnv(config=get_env_config(), data_loader=loader,
                       max_cont_agents=1, device='cuda',
                       action_type='continuous')
env.reset()
a = torch.zeros(8, 1, 3, device='cuda')
for _ in range(3):
    env.step_dynamics(a)
cur = list(env.data_batch)
print('STEP: 初始 8 个场景已跑通', flush=True)

# A) 同一批
env.swap_data_batch(data_batch=list(cur))
print('A_SAME_BATCH_OK', flush=True)
env.reset()
for _ in range(2):
    env.step_dynamics(a)
print('A_RESET_OK', flush=True)

# B) 8 个同一场景（重复但内容已知）
env.swap_data_batch(data_batch=[cur[0]] * 8)
print('B_DUP_SCENE_OK', flush=True)
env.reset()
print('B_RESET_OK', flush=True)
print('ALL_SWAP_WORKS')
