"""诊断：ScenePack 周车位置范围 + agent_feat 输出的异常来源。"""
import sys

import numpy as np

REPO = '/gpudrive/baselines/IDC-Bridge'
for p in (REPO, REPO + '/src'):
    sys.path.insert(0, p)

from env.planner_features import load_pack  # noqa: E402

P = '/data/data_json/training/tfrecord-00000-of-01000_100.json'
pk = load_pack(P)
print('agents:', len(pk.agents), ' ego_pos:', len(pk.ego_pos))
print('ego_pos 范围: x[%.1f, %.1f] y[%.1f, %.1f]' % (
    pk.ego_pos[:, 0].min(), pk.ego_pos[:, 0].max(),
    pk.ego_pos[:, 1].min(), pk.ego_pos[:, 1].max()))

for i, a in enumerate(pk.agents[:5]):
    pos = a['pos']
    print('agent%d pos 范围 x[%.1f,%.1f] y[%.1f,%.1f] len=%.1f wid=%.1f valid_sum=%d' % (
        i, pos[:, 0].min(), pos[:, 0].max(), pos[:, 1].min(), pos[:, 1].max(),
        a['length'], a['width'], int(np.asarray(a['valid']).sum())))

# 模拟 shift + agent_feat
m = np.array([pk.ego_pos[0, 0] - 1.8, pk.ego_pos[0, 1] - 9.2])   # 近似均值
pk.shift(-m[0], -m[1])
ego = pk.ego_pos[10]
ag, am = pk.agent_feat(10, (float(ego[0]), float(ego[1]), 0.0))
print('\nagent_feat 输出: shape', ag.shape, ' absmax %.1f' % float(np.abs(ag).max()),
      ' n=%d' % int(am.sum()))
worst = int(np.argmax(np.abs(ag).max(1)))
print('最差 agent 行:', np.round(ag[worst], 1))
print('  -> 前 8 维 = [rel_x, rel_y, cos, sin, v, len, wid, valid]')
