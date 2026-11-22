"""M1-1: 把 CARLA 各 Town 的 .xodr 解析成车道中心线折线，存成 npz。

不需要启动 CARLA 服务端：carla.Map(name, xodr_text) 可以离线构造。
输出：/workspace/data/carla_maps/map_<Town>.npz
      lanes: (N, 3)  [x, y, lane_width]
      lane_pts: 每条的折线点数
      lane_ids: (N,) [road_id*1000 + lane_id]
"""
import argparse
import glob
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--xodr-dirs', default='')
    ap.add_argument('--out', default='/workspace/data/carla_maps')
    ap.add_argument('--step', type=float, default=2.0)
    args = ap.parse_args()
    import carla

    dirs = [d for d in args.xodr_dirs.split(',') if d]
    files = []
    for d in dirs:
        files += glob.glob(os.path.join(d, '**', '*.xodr'), recursive=True)
    files = sorted(set(files))
    print(f'找到 {len(files)} 个 .xodr')
    os.makedirs(args.out, exist_ok=True)

    for fp in files:
        town = os.path.basename(fp).replace('.xodr', '')
        try:
            txt = open(fp).read()
            m = carla.Map(town, txt)
            wps = m.generate_waypoints(args.step)
        except Exception as e:
            print(f'{town}: 解析失败 {e}')
            continue
        groups = {}
        for wp in wps:
            key = (wp.road_id, wp.section_id, wp.lane_id)
            tf = wp.transform
            groups.setdefault(key, []).append(
                (tf.location.x, tf.location.y, wp.lane_width))
        polys, ids = [], []
        for key, pts in groups.items():
            if len(pts) < 2:
                continue
            polys.append(np.asarray(pts, np.float32))
            ids.append(key[0] * 1000 + key[2])
        if not polys:
            print(f'{town}: 无车道')
            continue
        lens = np.asarray([len(p) for p in polys], np.int32)
        flat = np.concatenate(polys, 0)
        out = os.path.join(args.out, f'map_{town}.npz')
        np.savez_compressed(out, lanes=flat, lane_pts=lens,
                            lane_ids=np.asarray(ids, np.int32))
        print(f'{town}: {len(polys)} 条车道, {len(flat)} 点 -> {os.path.basename(out)}')


if __name__ == '__main__':
    main()
