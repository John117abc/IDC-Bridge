"""D1 诊断：注入参考的速度 vs 日志专家速度（判断"速度失配"）。"""
import json
import os

import numpy as np

NPZ = '/workspace/data/head_plans2.npz'
JSON_DIR = '/data/data_json/training'

z = np.load(NPZ, allow_pickle=True)
names, paths, speeds = z['names'], z['paths'], z['speeds']
print(f"{'场景':28s} {'日志v0':>7s} {'参考v0':>7s} {'比值':>6s} "
      f"{'参考v20':>8s} {'参考v50':>8s}")
ratios = []
for i, n in enumerate(names):
    d = json.load(open(os.path.join(JSON_DIR, str(n))))
    idx = (d.get('metadata') or {}).get('sdc_track_index')
    pos = d['objects'][int(idx)]['position']
    v0 = float(np.linalg.norm(np.array([pos[1]['x'], pos[1]['y']]) -
                              np.array([pos[0]['x'], pos[0]['y']])) / 0.1)
    r0 = float(speeds[i][0][0])
    r20 = float(speeds[i][0][20])
    r50 = float(speeds[i][0][50])
    ratio = r0 / max(v0, 0.5)
    ratios.append(ratio)
    print(f"{str(n)[:28]:28s} {v0:7.2f} {r0:7.2f} {ratio:6.2f} "
          f"{r20:8.2f} {r50:8.2f}")

r = np.asarray(ratios)
print(f"\n比值: p50={np.median(r):.2f}  min={r.min():.2f}  max={r.max():.2f}  "
      f"|ratio-1|>0.5 的场景 {100*np.mean(np.abs(r-1) > 0.5):.0f}%")
