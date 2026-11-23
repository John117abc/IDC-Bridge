#!/bin/bash
set -u
C=idc_hist
CR=/gpudrive/baselines/IDC-Bridge
PY=/gpudrive/.venv/bin/python

docker exec $C bash -lc "cd $CR && $PY - <<'PY'
import json, sys, numpy as np
sys.path.insert(0, 'src')
p = '/data/data_json/training/tfrecord-00000-of-01000_100.json'
d = json.load(open(p))
r0 = d['roads'][0]
print('road keys:', list(r0.keys()))
g = r0['geometry'][0]
print('geometry[0] =', g, ' type:', type(g).__name__)
if isinstance(g, dict):
    print('  keys:', list(g.keys()))
# 模拟 ScenePack 的解析
A = []
for r in (d.get('roads') or []):
    typ = r.get('type')
    if typ not in ('lane', 'road_line', 'road_edge') or len(r.get('geometry') or []) < 2:
        continue
    try:
        L = np.asarray([[q['x'], q['y']] for q in r['geometry']], np.float64)
        A.append(L)
    except Exception as e:
        print('  解析异常:', type(e).__name__, e)
        break
print('可解析路段数:', len(A))

# 对象结构
o = d['objects'][0]
print('object keys:', list(o.keys())[:12])
pos = o.get('position')
print('position 类型:', type(pos).__name__, 'len', len(pos) if pos is not None else None)
if pos:
    print('position[0] =', pos[0], type(pos[0]).__name__)
PY" 2>&1 | tail -20
