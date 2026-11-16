"""从原始 Waymo tfrecord 导出历史帧（纯 Python TFRecord 读取 + 多进程）。

每个 scenario 导出成一条 npz：
    scenario_id, current_time_index, sdc_index
    track_ids  (A,)
    track_types(A,)
    length,width (A,)
    x,y,heading,vx,vy (A,91)  float32
    valid (A,91) bool
其中 A 是该场景的 agent 数，91 帧里 [0:current_time_index] 是历史。

用法：
    python tools/dump_history.py --out /workspace/data/history --shards 0:1
    python tools/dump_history.py --out /workspace/data/history --workers 8
"""
import argparse
import glob
import os
import struct
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

TREC = '/data/data/training/training.tfrecord-*'


def iter_records(path):
    """纯 Python 读 TFRecord（格式：uint64 len | crc32 len | data | crc32 data）。"""
    with open(path, 'rb') as f:
        while True:
            head = f.read(8)
            if len(head) < 8:
                return
            (n,) = struct.unpack('<Q', head)
            f.read(4)                       # crc of length
            data = f.read(n)
            f.read(4)                       # crc of data
            if len(data) < n:
                return
            yield data


def parse_scenario(raw):
    from waymo_open_dataset.protos import scenario_pb2
    sc = scenario_pb2.Scenario()
    sc.ParseFromString(raw)
    T = 91
    A = len(sc.tracks)
    x = np.zeros((A, T), np.float32)
    y = np.zeros((A, T), np.float32)
    hd = np.zeros((A, T), np.float32)
    vx = np.zeros((A, T), np.float32)
    vy = np.zeros((A, T), np.float32)
    valid = np.zeros((A, T), bool)
    ids = np.zeros(A, np.int64)
    types = np.zeros(A, np.int32)
    length = np.zeros(A, np.float32)
    width = np.zeros(A, np.float32)
    for a, tr in enumerate(sc.tracks):
        ids[a] = tr.id
        types[a] = int(tr.object_type)
        st = tr.states
        n = min(len(st), T)
        for j in range(n):
            s = st[j]
            x[a, j] = s.center_x
            y[a, j] = s.center_y
            hd[a, j] = s.heading
            vx[a, j] = s.velocity_x
            vy[a, j] = s.velocity_y
            valid[a, j] = s.valid
            if s.length:
                length[a] = s.length
                width[a] = s.width
    return dict(scenario_id=np.array(sc.scenario_id).astype('U'),
                current_time_index=np.int32(sc.current_time_index),
                sdc_index=np.int32(sc.sdc_track_index),
                track_ids=ids, track_types=types, length=length, width=width,
                x=x, y=y, heading=hd, vx=vx, vy=vy, valid=valid)


def process_shard(args):
    path, out_dir = args
    base = os.path.basename(path).replace('training.tfrecord-', 'shard_')
    out = os.path.join(out_dir, base + '.npz')
    if os.path.exists(out):
        return base, -1
    rows = []
    try:
        for raw in iter_records(path):
            try:
                rows.append(parse_scenario(raw))
            except Exception:
                continue
    except Exception as e:                       # noqa: BLE001
        return base, f'err {e}'
    if not rows:
        return base, 0
    # 变长 agent 数 → 拼接并记录起始下标
    counts = np.array([len(r['track_ids']) for r in rows], np.int32)
    starts = np.r_[0, np.cumsum(counts)[:-1]].astype(np.int32)
    cat = lambda k, dt: np.concatenate([r[k] for r in rows], 0).astype(dt)
    np.savez_compressed(
        out,
        scenario_id=np.array([str(r['scenario_id']) for r in rows]),
        current_time_index=np.array([int(r['current_time_index']) for r in rows],
                                    np.int32),
        sdc_index=np.array([int(r['sdc_index']) for r in rows], np.int32),
        counts=counts, starts=starts,
        track_ids=cat('track_ids', np.int64),
        track_types=cat('track_types', np.int32),
        length=cat('length', np.float32),
        width=cat('width', np.float32),
        x=cat('x', np.float32), y=cat('y', np.float32),
        heading=cat('heading', np.float32),
        vx=cat('vx', np.float32), vy=cat('vy', np.float32),
        valid=cat('valid', bool),
    )
    return base, len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--shards', default=None, help='如 0:5，默认全部')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    files = sorted(glob.glob(TREC))
    if args.shards:
        a, b = args.shards.split(':')
        files = files[int(a):int(b)]
    print(f'待处理 {len(files)} 个 shard，workers={args.workers}', flush=True)
    jobs = [(f, args.out) for f in files]
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for base, n in ex.map(process_shard, jobs):
            done += 1
            if done % 20 == 0 or not isinstance(n, int):
                print(f'[{done}/{len(files)}] {base} -> {n}', flush=True)
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
