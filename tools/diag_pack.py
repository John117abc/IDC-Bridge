"""隔离诊断：env.data_loader.dataset[0] 到底是什么、ScenePack 能否解析。"""
import os
import sys

import numpy as np

REPO = '/gpudrive/baselines/IDC-Bridge'
for p in (REPO, os.path.join(REPO, 'src')):
    sys.path.insert(0, p)

from env.env_utils import get_env_config  # noqa: E402
from env.planner_features import load_pack  # noqa: E402
from gpudrive.env.env_torch import GPUDriveTorchEnv  # noqa: E402
from gpudrive.env.dataset import SceneDataLoader  # noqa: E402

DATA = '/data/data_json/training'
cfg = get_env_config()
loader = SceneDataLoader(root=DATA, batch_size=1, dataset_size=1,
                         sample_with_replacement=False, shuffle=False, seed=0)
print('loader 数据集大小:', len(loader.dataset))
entry = loader.dataset[0]
print('dataset[0] 类型:', type(entry).__name__)
print('dataset[0] 值:', str(entry)[:200])

env = GPUDriveTorchEnv(config=cfg, data_loader=loader, max_cont_agents=8,
                       device='cuda', action_type='continuous')
print('env 建好')

try:
    pk = load_pack(str(entry))
    print('ScenePack: seg_a =', len(pk.seg_a), ' agents =', len(pk.agents),
          ' ego_pos 长度 =', len(pk.ego_pos))
    if len(pk.agents):
        print('  首车位置示例:', np.round(pk.agents[0]['pos'][:2], 1))
except Exception as e:
    print('ScenePack 失败:', type(e).__name__, e)

print('map_observation tensor 形状:',
      tuple(env.sim.map_observation_tensor().shape))
print('absolute_self_observation 形状:',
      tuple(env.sim.absolute_self_observation_tensor().shape))
