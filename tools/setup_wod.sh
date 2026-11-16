#!/usr/bin/env bash
P=/gpudrive/.venv/bin/python
echo "=== 装 waymo-open-dataset-tf-2-12-0（不装依赖）==="
$P -m pip install --no-deps -i https://pypi.org/simple \
    waymo-open-dataset-tf-2-12-0 2>&1 | grep -vE 'notice|WARNING' | tail -3
echo "=== 测 import ==="
$P -c "from waymo_open_dataset.protos import scenario_pb2; print('pb2 OK')" 2>&1 | tail -6
echo "=== 读一条 tfrecord 试试 ==="
$P - <<'PY' 2>&1 | tail -8
import glob
try:
    import tensorflow as tf
    from waymo_open_dataset.protos import scenario_pb2
    f = sorted(glob.glob('/data/data/training/*.tfrecord-*'))[0]
    ds = tf.data.TFRecordDataset([f])
    for raw in ds.take(1):
        sc = scenario_pb2.Scenario()
        sc.ParseFromString(raw.numpy())
        print('scenario_id', sc.scenario_id)
        print('current_time_index', sc.current_time_index)
        print('tracks', len(sc.tracks))
        if sc.tracks:
            print('states of track0', len(sc.tracks[0].states))
except Exception as e:
    print('FAIL:', type(e).__name__, str(e)[:200])
PY
