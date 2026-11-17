"""诊断：8-waypoint 锚点 densify 后到底长什么样，为什么 oracle 归零。"""
import glob
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "tools"))

from metrics.openloop_pdms import Scene, score_trajectory  # noqa: E402
from eval_head_pdms import densify, ego_to_world  # noqa: E402

A = np.load('/workspace/data/head_dd/anchors.npy')
print('anchors', A.shape)
print('每条锚点 终点 x(前向) y(横向):')
for k in range(min(20, A.shape[0])):
    print(f'  {k:2d}: wp={np.round(A[k], 1).tolist()}')

files = sorted(glob.glob('/data/data_json/training/tfrecord-*.json'))[:5]
for fp in files:
    d = json.load(open(fp))
    sc = Scene(d, horizon=40)
    p0, h0 = sc.expert_pos[sc.t_now], float(sc.expert_heading[sc.t_now])
    gt = sc.expert_future[:40]
    print(f'\n场景 {os.path.basename(fp)[-12:]} n_future={sc.n_future} '
          f'p0={np.round(p0,1)} h0={h0:.2f}')
    print('  GT 终点(世界)', np.round(gt[-1], 1),
          ' 距离', round(float(np.linalg.norm(gt[-1]-p0)), 1), 'm')
    ex = score_trajectory(sc, gt)
    print('  expert:', {k: (round(v, 2) if isinstance(v, float) else v)
                        for k, v in ex.items() if k in
                        ('driving_score', 'nc', 'dac', 'ddc', 'ep')})
    best = None
    for k in range(A.shape[0]):
        w = ego_to_world(densify(A[k], 40), p0, h0)
        r = score_trajectory(sc, w)
        if best is None or r['driving_score'] > best[1]['driving_score']:
            best = (k, r, w)
    k, r, w = best
    print(f'  最佳锚点 {k}:', {kk: (round(v, 2) if isinstance(v, float) else v)
                              for kk, v in r.items() if kk in
                              ('driving_score', 'nc', 'dac', 'ddc', 'ep')})
    print('  该锚点终点(世界)', np.round(w[-1], 1),
          ' 与 GT 差', round(float(np.linalg.norm(w[-1]-gt[-1])), 2), 'm')
