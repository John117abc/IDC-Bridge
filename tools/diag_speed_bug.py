"""验证：候选路径的 speed 是否与点间距/时间基准自洽（2.3 m/s 是不是 bug）。"""
import os
import sys

import numpy as np

REPO = '/gpudrive/baselines/IDC-Bridge'
for p in (REPO, os.path.join(REPO, 'src')):
    sys.path.insert(0, p)

from env.env_utils import get_env_config  # noqa: E402
from gpudrive.env.env_torch import GPUDriveTorchEnv  # noqa: E402
from gpudrive.env.dataset import SceneDataLoader  # noqa: E402
from env.idc_state_builder import GPUDriveObservationBuilder  # noqa: E402
from env.env_utils import load_scenes, get_ego_indices  # noqa: E402

DATA = '/data/data_json/training'
cfg = get_env_config()
loader = SceneDataLoader(root=DATA, batch_size=2, dataset_size=2,
                         sample_with_replacement=False, shuffle=False, seed=0)
env = GPUDriveTorchEnv(config=cfg, data_loader=loader, max_cont_agents=8,
                       device='cuda', action_type='continuous')
scenes = load_scenes(env)
b = GPUDriveObservationBuilder(env, scenes)
ego_idx = get_ego_indices(env, 2)
b.generate_candidate_paths(ego_idx, num_paths=1)

dt = float(getattr(cfg, 'dt', 0.1))
print('cfg.dt =', dt)
for w in range(2):
    a = int(ego_idx[w])
    paths = b.candidate_paths[w][a]
    p = paths[0]
    pos = np.asarray(p['pos'], np.float64)
    spd = np.asarray(p['speed'], np.float64)
    d = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    print(f'\n--- world {w}: 候选0 ---')
    print('  点数', len(pos), ' speed 点数', len(spd),
          ' last_valid_idx', p.get('last_valid_idx', '-'))
    print('  相邻点距离: 中位 %.3f 最大 %.3f m' % (np.median(d), d.max()))
    print('  speed 数组: 中位 %.3f 最小 %.3f 最大 %.3f m/s'
          % (np.median(spd), spd.min(), spd.max()))
    print('  自洽性: speed*dt 中位 %.3f  vs  点间距中位 %.3f'
          % (np.median(spd) * dt, np.median(d)))
    print('  整条路径长度 %.1f m' % d.sum())
    print('  若按 speed 走完需 %.1f s' % (d.sum() / max(np.median(spd), 1e-3)))
    print('  若按 91*dt 走完需 %.1f s' % (len(pos) * dt))
    print('  pos[0]=%s pos[10]=%s pos[40]=%s pos[-1]=%s'
          % (np.round(pos[0], 1), np.round(pos[min(10, len(pos)-1)], 1),
             np.round(pos[min(40, len(pos)-1)], 1), np.round(pos[-1], 1)))
