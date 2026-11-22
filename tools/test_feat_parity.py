"""校验 planner_features.ScenePack 与 build_head_dataset.build_one 逐位一致。"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, '/gpudrive/baselines/IDC-Bridge/src')
sys.path.insert(0, '/gpudrive/baselines/IDC-Bridge/tools')

from build_head_dataset import build_one  # noqa: E402
from env.planner_features import load_pack  # noqa: E402


def main():
    z = np.load('/workspace/data/rt_feat.npz', allow_pickle=True)
    names = [str(z[f'name{w}']) for w in range(4) if f'name{w}' in z.files]
    d_dir = '/data/data_json/training'
    for name in names:
        fp = os.path.join(d_dir, name)
        if not os.path.exists(fp):
            continue
        d = json.load(open(fp))
        ref = build_one(d, np.zeros((6, 8, 2), np.float32))
        pack = load_pack(fp)
        t = 10
        _p, _h = pack.ego_pose(t)
        pose = (float(_p[0]), float(_p[1]), float(_h))
        e = pack.ego_feat(t)
        ag, am = pack.agent_feat(t, pose)
        mp, mm = pack.map_feat(pose)
        de = np.abs(e - ref['ego']).max()
        ag_ref, am_ref = ref['agent'], ref['amask']
        n_ref = int((am_ref > 0.5).sum())
        n = int((am > 0.5).sum())
        d_ag = np.abs(ag[:n] - ag_ref[:n]).max() if n and n == n_ref else float('nan')
        mp_ref, mm_ref = ref['map'], ref['mmask']
        d_mp = np.abs(mp - mp_ref).max()
        print('%-34s ego %.2e | agent %.2e (%d/%d) | map %.2e'
              % (name, de, d_ag, n, n_ref, d_mp))


if __name__ == '__main__':
    main()

