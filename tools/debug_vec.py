"""调试：geo_from_npz + score_batch 的分项输出。"""
import sys

import numpy as np

sys.path.insert(0, '/gpudrive/baselines/IDC-Bridge/src')
sys.path.insert(0, '/gpudrive/baselines/IDC-Bridge/tools')

from metrics.openloop_pdms import Scene, score_trajectory  # noqa
from metrics.openloop_pdms_vec import geo_from_npz, score_batch  # noqa
from validate_pdms_vec import make_cands  # noqa
import json, os  # noqa

data = sys.argv[1] if len(sys.argv) > 1 else '/workspace/data/planner300/train.npz'
z = np.load(data, allow_pickle=True)
rng = np.random.default_rng(0)

for i in range(3):
    g = geo_from_npz(z, i)
    print('--- 样本 %d (%s) ---' % (i, z['scene_id'][i]))
    print('  周车数 %d  走廊线段 %d (lane %d, edge %d)  T=%d'
          % (g.ap.shape[0], len(g.ea) + len(g.la), len(g.la), len(g.ea), g.n_future))
    print('  hm: coll %d off %d ddc %d  ttc 有限 %d/%d  lat p50 %.2f'
          % (g.hm['coll'].sum(), g.hm['off'].sum(), g.hm['ddc'].sum(),
             np.isfinite(g.hm['ttc']).sum(), len(g.hm['ttc']),
             np.median(g.hm['lat'])))
    cands = make_cands(z['gt'][i], rng, 0.5, 4)
    o = score_batch(g, cands)
    print('  PDMS %s' % np.round(o['driving_score'], 2))
    print('  nc %s dac %s ddc %s' % (np.round(o['nc'], 0), np.round(o['dac'], 0),
                                     np.round(o['ddc'], 0)))
    print('  ep %s' % np.round(o['ep'], 3))
    print('  ttc %s' % np.round(o['ttc'], 3))
    print('  comfort %s' % np.round(o['comfort'], 3))
    print('  lk %s' % np.round(o['lk'], 3))
    print('  coll 步数 %s' % o['coll'].sum(1))
    print('  off 步数 %s' % o['off'].sum(1))
    print('  raw_ttc[0][:5] %s' % np.round(o['raw_ttc'][0][:5], 2))

    fp = os.path.join('/data/data_json/training', str(z['scene_id'][i]))
    if os.path.exists(fp):
        sc = Scene(json.load(open(fp)), horizon=40)
        c, s = np.cos(z['ego_world'][i][2]), np.sin(z['ego_world'][i][2])
        R = np.array([[c, -s], [s, c]])
        cw = cands @ R.T + z['ego_world'][i][:2][None]
        ex = [score_trajectory(sc, x) for x in cw]
        print('  exact PDMS %s' % np.round([e['driving_score'] for e in ex], 2))
        print('  exact ep %s' % np.round([e['ep'] for e in ex], 3))
        print('  exact ttc %s' % np.round([e['ttc'] for e in ex], 3))
        print('  exact lk %s' % np.round([e['lk'] for e in ex], 3))
