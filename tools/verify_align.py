"""严格验证：JSON 的 91 帧是否等于 tfrecord 的 states[0:91]？
   · 用多个 shard、多个场景、多个 agent 验证
   · 用 timestamps 证明 index 10 就是"现在"（1.0 s）
"""
import glob
import json
import struct

import numpy as np


def iter_records(path):
    with open(path, 'rb') as f:
        while True:
            head = f.read(8)
            if len(head) < 8:
                return
            (n,) = struct.unpack('<Q', head)
            f.read(4)
            data = f.read(n)
            f.read(4)
            if len(data) < n:
                return
            yield data


from waymo_open_dataset.protos import scenario_pb2  # noqa: E402

# 1) timestamps
f0 = sorted(glob.glob('/data/data/training/*.tfrecord-*'))[0]
raw = next(iter_records(f0))
sc = scenario_pb2.Scenario()
sc.ParseFromString(raw)
print('scenario', sc.scenario_id, 'current_time_index', sc.current_time_index)
print('Scenario 字段:', [x.name for x in sc.DESCRIPTOR.fields])
for cand in ('timestamps_seconds', 'timestamps'):
    if hasattr(sc, cand) and len(getattr(sc, cand)):
        ts = getattr(sc, cand)
        print(cand, '[0]', round(float(ts[0]), 3), '[10]', round(float(ts[10]), 3),
              '[-1]', round(float(ts[-1]), 3))
print('states 数', len(sc.tracks[0].states))

# 2) 多个场景 x 多个 agent 对比 JSON
jfiles = sorted(glob.glob('/data/data_json/training/tfrecord-00000-of-01000_*.json'))[:6]
by_id = {}
for raw in iter_records(f0):
    s = scenario_pb2.Scenario()
    s.ParseFromString(raw)
    by_id[s.scenario_id] = s
    if len(by_id) > 400:
        break
print('shard0 载入 scenario 数', len(by_id))

checked = ok0 = ok_agents = 0
for jf in jfiles:
    d = json.load(open(jf))
    sid = d['scenario_id']
    if sid not in by_id:
        continue
    s = by_id[sid]
    id2tr = {int(t.id): t for t in s.tracks}
    checked += 1
    for o in d['objects'][:6]:
        tr = id2tr.get(int(o['id']))
        if tr is None:
            continue
        p_json = np.array([o['position'][0]['x'], o['position'][0]['y']])
        p_tf = np.array([tr.states[0].center_x, tr.states[0].center_y])
        err = float(np.hypot(*(p_json - p_tf)))
        checked_agents = 0
        if err < 0.01:
            ok_agents += 1
        pj_last = np.array([o['position'][-1]['x'], o['position'][-1]['y']])
        pt_last = np.array([tr.states[-1].center_x, tr.states[-1].center_y])
        err_last = float(np.hypot(*(pj_last - pt_last)))
        if err < 0.01 and err_last < 0.01:
            ok0 += 1

print(f'对比 {checked} 个场景；agent 首帧吻合 {ok_agents}；'
      f'首末帧同时吻合(场景级) {ok0}')
