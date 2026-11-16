"""验证 JSON 与 tfrecord 的对齐：JSON 的 position[0] 对应 states[0] 还是 states[10]？"""
import glob
import json

import numpy as np

# 1) 取一个 JSON
jf = sorted(glob.glob('/data/data_json/training/tfrecord-00000-of-01000_*.json'))[0]
d = json.load(open(jf))
sid = d['scenario_id']
meta = d.get('metadata') or {}
sdc = d['objects'][meta['sdc_track_index']]
p0 = np.array([sdc['position'][0]['x'], sdc['position'][0]['y']])
p1 = np.array([sdc['position'][1]['x'], sdc['position'][1]['y']])
plast = np.array([sdc['position'][-1]['x'], sdc['position'][-1]['y']])
print('JSON:', jf.split('/')[-1])
print('scenario_id:', sid, ' | sdc id:', sdc.get('id'),
      ' | 帧数:', len(sdc['position']))
print(f'JSON pos[0]=({p0[0]:.2f},{p0[1]:.2f}) pos[-1]=({plast[0]:.2f},{plast[1]:.2f})')

# 2) 在 shard_00000 里找这个 scenario
z = np.load('/workspace/data/history/shard_00000-of-01000.npz', allow_pickle=True)
sids = [str(s) for s in z['scenario_id']]
if sid not in sids:
    print('该 scenario 不在 shard0（属于别的分片），换一个 JSON 再试')
    raise SystemExit(0)
i = sids.index(sid)
st, cnt = int(z['starts'][i]), int(z['counts'][i])
ids = z['track_ids'][st:st + cnt]
j = int(np.where(ids == sdc['id'])[0][0]) if (ids == sdc['id']).any() else 0
X = z['x'][st + j]
Y = z['y'][st + j]
print('tfrecord: sdc index', j, ' current_time_index', int(z['current_time_index'][i]))
for k in (0, 5, 10):
    d0 = np.hypot(X[k] - p0[0], Y[k] - p0[1])
    print(f'  |states[{k}] - JSON[0]| = {d0:.3f} m   ({X[k]:.2f},{Y[k]:.2f})')
for k in (-1, -2, -11):
    dk = np.hypot(X[k] - plast[0], Y[k] - plast[1])
    print(f'  |states[{k}] - JSON[-1]| = {dk:.3f} m')
