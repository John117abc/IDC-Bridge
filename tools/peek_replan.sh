#!/bin/bash
# 看导出的 replan npz 里第 0 个世界的第 0 帧路径到底是什么
cd /gpudrive/baselines/IDC-Bridge || exit 1
/gpudrive/.venv/bin/python - <<'PY'
import numpy as np
z = np.load('/workspace/data/replan_gt0.npz', allow_pickle=True)
print('frames', z['frames'], 'names', z['names'][:2])
p = z['paths'][0, 0, 0]          # world0, frame0, cand0
s = z['speeds'][0, 0, 0]
d = np.linalg.norm(np.diff(p, axis=0), axis=1)
print('前 6 点:', np.round(p[:6], 2).tolist())
print('相邻步长前 6:', np.round(d[:6], 3).tolist())
print('总长 %.1f m, 平均步长 %.3f m, 最大 %.3f' % (d.sum(), d.mean(), d.max()))
print('speed 前 6:', np.round(s[:6], 2).tolist())
print('origin_pos', z['origin_pos'][0], 'origin_heading', z['origin_heading'][0])
PY
