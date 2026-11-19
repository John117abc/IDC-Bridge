"""§6.1 验收：外部注入的 road_dist 是否与地图候选一致。"""
import glob
import json
import sys

import numpy as np

sys.path.insert(0, '/gpudrive/baselines/IDC-Bridge/src')
from env.candidate_sampler import SamplerConfig, boundary_segments, dist_to_segments  # noqa
from env.candidate_sampler import road_dist_for_path  # noqa
from env.map_candidates import generate_map_candidates  # noqa

FILES = sorted(glob.glob('/data/data_json/training/tfrecord-*.json'))[:5]
cfg = SamplerConfig()

print(f'{"scene":28s} {"K":>3s} {"max|diff|":>10s} {"n_inf":>6s} {"has_edge":>8s}')
worst = 0.0
for fp in FILES:
    d = json.load(open(fp))
    objs = d['objects']
    sdc = int(d['metadata']['sdc_track_index'])
    p = np.asarray([[q['x'], q['y']] for q in objs[sdc]['position']], np.float64)
    hh = np.asarray(objs[sdc]['heading'], np.float64)
    t = 10
    pose = (p[t, 0], p[t, 1], hh[t])

    # 环境地图行（与 GPUDrive 的 map_observation_tensor 同格式）
    map_rows = []
    for r in d['roads']:
        _g = r['geometry']
        pts = (np.asarray([[q['x'], q['y']] for q in _g], np.float64)
               if isinstance(_g, list) and _g and isinstance(_g[0], dict)
               else np.asarray(_g, np.float64))
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            mid = 0.5 * (a + b)
            L = float(np.linalg.norm(b - a))
            th = float(np.arctan2((b - a)[1], (b - a)[0]))
            typ = 1 if r.get('type') == 'road_edge' else 2
            map_rows.append([mid[0], mid[1], L, 0.0, 0.0, th, typ])
    map_rows = np.asarray(map_rows, np.float64) if map_rows else np.zeros((0, 7))

    cands = generate_map_candidates(pose, map_rows,
                                    offsets=(0.0,), cfg=cfg,
                                    speed0=8.0, n=91)
    flat = []
    for item in cands:
        if isinstance(item, (list, tuple)):
            flat.extend(item)
        else:
            flat.append(item)
    cands = [x for x in flat if hasattr(x, 'pos')]
    if not cands:
        print('  (no candidates)')
        continue

    a_seg, b_seg = boundary_segments(map_rows)
    has_edge = len(a_seg) > 0
    diffs = []
    n_inf = 0
    for c in cands:
        ref = np.asarray(c.clearance, np.float64)
        # 走"外部注入同款"路径
        mine = dist_to_segments(c.pos, a_seg, b_seg) if has_edge else np.full(len(c.pos), np.inf)
        fin = np.isfinite(ref) & np.isfinite(mine)
        n_inf += int((~np.isfinite(mine)).sum())
        if fin.any():
            diffs.append(float(np.max(np.abs(ref[fin] - mine[fin]))))
    mx = max(diffs) if diffs else float('nan')
    worst = max(worst, 0.0 if not np.isfinite(mx) else mx)
    print(f'{fp.split("/")[-1][:28]:28s} {len(cands):3d} {mx:10.6f} {n_inf:6d} {str(has_edge):>8s}')

print(f'\n最大逐点差异 = {worst:.3e} m（应在浮点误差内，<1e-4）')
