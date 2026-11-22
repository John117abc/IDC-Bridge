"""探测：奖励几何需要的缓冲区尺寸 + build_one 单场景耗时。"""
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, '/gpudrive/baselines/IDC-Bridge/src')
sys.path.insert(0, '/gpudrive/baselines/IDC-Bridge/tools')

files = sorted(glob.glob('/data/data_json/training/tfrecord-*.json'))
rng = np.random.default_rng(0)
files = [files[i] for i in rng.choice(len(files), 30, replace=False)]

rows = []
t_load = t_scene = 0.0
for fp in files:
    t0 = time.time()
    d = json.load(open(fp))
    t_load += time.time() - t0
    objs = d.get('objects') or []
    idx = (d.get('metadata') or {}).get('sdc_track_index')
    sdc = objs[int(idx)]
    p0 = np.array([sdc['position'][10]['x'], sdc['position'][10]['y']])

    ne = {r: 0 for r in (20, 30, 45, 60, 80)}
    nl = {r: 0 for r in (20, 30, 45, 60, 80)}
    for r in (d.get('roads') or []):
        typ = r.get('type')
        if typ not in ('lane', 'road_line', 'road_edge') or len(r.get('geometry') or []) < 2:
            continue
        L = np.asarray([[q['x'], q['y']] for q in r['geometry']], np.float64)
        mid = 0.5 * (L[:-1] + L[1:])
        dist = np.linalg.norm(mid - p0[None], axis=1)
        for rad in ne:
            n = int((dist < rad).sum())
            if typ == 'road_edge':
                ne[rad] = max(ne[rad], 0) + n
            else:
                nl[rad] = max(nl[rad], 0) + n
    rows.append((len(objs), ne, nl))

print('场景 %d' % len(rows))
print('目标数: p50=%.0f p90=%.0f max=%s' % (
    np.median([r[0] for r in rows]), np.percentile([r[0] for r in rows], 90),
    max(r[0] for r in rows)))
for rad in (20, 30, 45, 60, 80):
    e = np.array([r[1][rad] for r in rows])
    l = np.array([r[2][rad] for r in rows])
    print('半径 %2d m: edge p50=%4.0f p90=%4.0f max=%4d | lane p50=%4.0f p90=%4.0f max=%4d'
          % (rad, np.median(e), np.percentile(e, 90), e.max(),
             np.median(l), np.percentile(l, 90), l.max()))
print('json.load 平均 %.1f ms/场景' % (1000 * t_load / len(rows)))

# build_one 耗时（含 Scene + step_metrics）
from build_head_dataset import build_one  # noqa
anc = np.zeros((20, 8, 2), np.float32)
t0 = time.time()
for fp in files[:10]:
    d = json.load(open(fp))
    build_one(d, anc)
dt = (time.time() - t0) / 10
print('build_one 平均 %.0f ms/场景（含 json 读取则 %.0f ms）'
      % (1000 * dt, 1000 * (dt + t_load / len(rows))))
