#!/bin/bash
# 定位：导出的 GT 路径为什么几乎不动
cd /gpudrive/baselines/IDC-Bridge || exit 1
/gpudrive/.venv/bin/python - <<'PY'
import json, sys
import numpy as np
sys.path.insert(0, 'src'); sys.path.insert(0, 'tools')
from eval_head_pdms import densify, ego_to_world

f = '/data/data_json/training/tfrecord-00000-of-01000_154.json'
d = json.load(open(f))
idx = int(d['metadata']['sdc_track_index'])
pos = np.asarray([[q['x'], q['y']] for q in d['objects'][idx]['position']], np.float64)
head = np.asarray(d['objects'][idx].get('heading') or [0.0]*len(pos))
t = 10
p0, h0 = pos[t], float(head[t])
print('len(pos)', len(pos), 'p0', np.round(p0, 2), 'h0 %.3f' % h0)
idx8 = [min(t + 5*(i+1), len(pos)-1) for i in range(8)]
fut = pos[idx8]
print('fut[0], fut[-1]:', np.round(fut[0], 2), np.round(fut[-1], 2))
rel = fut - p0
c, s = np.cos(-h0), np.sin(-h0)
rel = np.stack([rel[:, 0]*c - rel[:, 1]*s, rel[:, 0]*s + rel[:, 1]*c], 1)
print('rel[-1] (ego frame):', np.round(rel[-1], 2), ' 弧长 %.2f' % np.linalg.norm(np.diff(np.vstack([[0,0], rel]), axis=0), axis=1).sum())
p40 = densify(rel, 40)
print('p40[-1]:', np.round(p40[-1], 2), ' p40[10]:', np.round(p40[10], 2))
dw = ego_to_world(densify(p40, 40), p0, h0)
print('dw[-1]:', np.round(dw[-1], 2), ' 终点位移 %.2f m' % np.linalg.norm(dw[-1]-p0))
dw2 = ego_to_world(p40, p0, h0)
print('dw2[-1]:', np.round(dw2[-1], 2), ' 终点位移 %.2f m' % np.linalg.norm(dw2[-1]-p0))
PY
