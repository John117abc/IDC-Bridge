"""M1 合并版：一条 route 目录 → 我们的数据集样本（ego/agent/map/route/gt）。

输出 npz 字段：
  ego (T,23) agent (T,32,28) amask (T,32) map (T,128,7) mmask (T,128)
  route (T,24,4) [x, y, s, heading_delta]（ego 系） gt (T,8,2)
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_vector_dataset import build as build_frames          # noqa: E402
from build_route_carla import build_route, ego_track            # noqa: E402

N_ROUTE, ROUTE_STEP = 24, 2.0


def route_feat(route_world, s, pose):
    """把世界坐标 route 转到该帧 ego 系： [x, y, s, heading_delta]。"""
    x0, y0, h0 = pose
    d = route_world - np.array([x0, y0])
    c, sn = np.cos(-h0), np.sin(-h0)
    ex = np.stack([d[:, 0] * c - d[:, 1] * sn,
                   d[:, 0] * sn + d[:, 1] * c], 1)
    seg = np.diff(route_world, axis=0)
    hd = np.arctan2(seg[:, 1], seg[:, 0]) - h0
    hd = np.r_[hd, hd[-1] if len(hd) else 0.0]
    out = np.zeros((N_ROUTE, 4), np.float32)
    n = min(len(ex), N_ROUTE)
    out[:n, :2] = ex[:n]
    out[:n, 2] = s[:n]
    out[:n, 3] = np.cos(hd[:n])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root',
                    default='/mnt/2T_HDD/PlanT2.0/PlanT2_Dataset/data/'
                            'PlanT2_DS/PlanT_2_dataset/data')
    ap.add_argument('--maps', default='/mnt/2T_HDD/plant2_vector/maps')
    ap.add_argument('--out', default='/mnt/2T_HDD/plant2_vector/vector_ds')
    ap.add_argument('--scenarios', default='HighwayCutIn,'
                    'SignalizedJunctionRightTurn,NonSignalizedJunctionRightTurn')
    ap.add_argument('--max-routes', type=int, default=3)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    scs = ([d for d in sorted(os.listdir(args.data_root))
            if os.path.isdir(os.path.join(args.data_root, d))]
           if args.scenarios == 'all' else args.scenarios.split(','))
    for sc in scs:
        if not sc:
            continue
        _r = sorted(glob.glob(os.path.join(args.data_root, sc, 'Town*')) )
        routes = _r if args.max_routes <= 0 else _r[:args.max_routes]
        if not routes:
            print(f'{sc}: 无 route 目录'); continue
        agg = {k: [] for k in ('ego', 'agent', 'amask', 'map', 'mmask', 'route', 'gt')}
        n_ok = n_bad = 0
        for rd in routes:
            town = os.path.basename(rd).split('_')[0]
            mp = os.path.join(args.maps, f'map_{town}.npz')
            if not os.path.exists(mp):
                print(f'  {town}: 无地图'); continue
            try:
                samp = build_frames(rd, mp)
                trk = ego_track(rd)
                res = build_route(trk, mp)
            except Exception as e:
                print(f'  {os.path.basename(rd)}: 失败 {e}'); n_bad += 1; continue
            if not samp or res is None:
                n_bad += 1; continue
            rw = res[0]
            seg = np.linalg.norm(np.diff(rw, axis=0), axis=1)
            s = np.r_[0.0, np.cumsum(seg)]
            for i, d in enumerate(samp):
                pose = trk[min(i + 10, len(trk) - 1)]
                d['route'] = route_feat(rw, s, pose)
            for k in agg:
                agg[k].extend([d[k] for d in samp])
            n_ok += 1
        if n_ok == 0:
            print(f'{sc}: 全部失败'); continue
        out = {k: np.stack(v).astype(np.float32) for k, v in agg.items()}
        fp = os.path.join(args.out, f'{sc}.npz')
        np.savez_compressed(fp, **out)
        T = len(out['gt'])
        print(f'{sc}: routes {n_ok}/{n_ok+n_bad}  样本 {T}  '
              f'agent {out["amask"].sum(1).mean():.1f}  '
              f'map {out["mmask"].sum(1).mean():.1f}  '
              f'route非零 {(np.abs(out["route"][..., 0]).sum(1) > 0).mean():.0%}  '
              f'-> {os.path.basename(fp)}')
        print(f'   gt[0]={np.round(out["gt"][0, 0], 2).tolist()}  '
              f'gt[-1]={np.round(out["gt"][-1, -1], 2).tolist()}')


if __name__ == '__main__':
    main()
