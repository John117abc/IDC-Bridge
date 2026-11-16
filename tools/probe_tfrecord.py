"""探针：tfrecord 里能不能解析出历史帧？

检查可用的解析库，并读取一个原始 tfrecord 场景，打印：
  scenario_id / current_time_index / track 数 / 每条 track 的 state 数
"""
import glob
import sys

LIBS = {}
for name in ('tensorflow', 'waymo_open_dataset', 'tfrecord'):
    try:
        __import__(name)
        LIBS[name] = 'ok'
    except Exception as e:                       # noqa: BLE001
        LIBS[name] = f'missing: {type(e).__name__}'
print('libs:', LIBS)

files = sorted(glob.glob('/data/data/training/*.tfrecord-*'))
print('tfrecord 文件数:', len(files))
if not files:
    sys.exit('没有 tfrecord')

if LIBS['tensorflow'] != 'ok':
    sys.exit('没有 tensorflow，无法直接读 tfrecord')

import tensorflow as tf  # noqa: E402

path = files[0]
print('读第一个文件:', path)
ds = tf.data.TFRecordDataset([path])
try:
    from waymo_open_dataset.protos import scenario_pb2
    for raw in ds.take(1):
        sc = scenario_pb2.Scenario()
        sc.ParseFromString(raw.numpy())
        print(f'scenario_id={sc.scenario_id}')
        print(f'current_time_index={sc.current_time_index}')
        print(f'tracks={len(sc.tracks)}')
        if sc.tracks:
            st = sc.tracks[0].states
            print(f'track[0] id={sc.tracks[0].id} states={len(st)}')
            if st:
                print(f'  state[0] center=({st[0].center_x:.2f},{st[0].center_y:.2f}) '
                      f'valid={st[0].valid}')
                print(f'  state[-1] center=({st[-1].center_x:.2f},{st[-1].center_y:.2f}) '
                      f'valid={st[-1].valid}')
except Exception as e:                           # noqa: BLE001
    print('解析 scenario 失败:', type(e).__name__, e)
