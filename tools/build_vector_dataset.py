"""M1-2 smoke: 从 PlanT2 数据集的一条 route 目录构建我们的向量样本。

输出每帧：ego(23) agent(32,28) agent_mask(32) map(128,7) map_mask(128) gt(8,2)
用法：python build_vector_dataset.py --route <dir> --map <map_TownXX.npz> --out <npz>
"""
import argparse
import glob
import gzip
import json
import os

import numpy as np

HIST, N_MAX, M_MAX, T_FUT, STEP = 10, 32, 128, 8, 5
MAP_R, T_NOW = 30.0, 10


def _load(fp):
    with gzip.open(fp, 'rt') as fh:
        return json.load(fh)


def _yaw_xy(b):
    m = np.asarray(b['matrix'], np.float32)
    return float(b['wx']), float(b['wy']), b['wyaw'], m


def _norm(boxes):
    for b in boxes:
        m = np.asarray(b['matrix'], np.float64)
        b['wx'] = float(m[0][3])
        b['wy'] = float(m[1][3])
        b['wyaw'] = float(np.arctan2(m[1][0], m[0][0]))
    return boxes


def _to_ego(P, p0, h0):
    d = np.asarray(P, np.float64) - np.asarray(p0, np.float64)
    c, s = np.cos(-h0), np.sin(-h0)
    return np.stack([d[..., 0] * c - d[..., 1] * s,
                     d[..., 0] * s + d[..., 1] * c], -1)


def build(route_dir, map_npz):
    bf = sorted(glob.glob(os.path.join(route_dir, 'boxes', '*.json.gz')))
    mf = sorted(glob.glob(os.path.join(route_dir, 'measurements', '*.json.gz')))
    n = min(len(bf), len(mf))
    if n < T_NOW + 50:
        print(f'帧数不足: {n}')
        return None
    z = np.load(map_npz)
    lanes, lpts, lids = z['lanes'], z['lane_pts'], z['lane_ids']
    segA, segB, segW, segId = [], [], [], []
    o = 0
    for k, np_ in enumerate(lpts):
        L = lanes[o:o + np_]
        o += np_
        if len(L) < 2:
            continue
        segA.append(L[:-1, :2])
        segB.append(L[1:, :2])
        segW.append(L[:-1, 2])
        segId.append(np.full(len(L) - 1, lids[k]))
    A = np.concatenate(segA)
    B = np.concatenate(segB)
    W = np.concatenate(segW)

    frames = []
    id2pos = {}
    for f in range(n):
        boxes = _norm(_load(bf[f]))
        ego = [b for b in boxes if b['class'] == 'ego_car']
        if not ego:
            continue
        frames.append((f, boxes, ego[0]))
        for _b in boxes:
            _id = _b.get('id')
            if _id is None:
                continue
            id2pos.setdefault(_id, {})[f] = (_b['wx'], _b['wy'])
    if len(frames) < T_NOW + 50:
        return None

    out = []
    for i in range(T_NOW, len(frames) - T_FUT * STEP):
        p0 = np.array([frames[i][2]['wx'], frames[i][2]['wy']])
        h0 = frames[i][2]['wyaw']
        # ego 历史（世界→ego 系，含速度/角速度）
        hist = []
        for j in range(i - HIST + 1, i + 1):
            b = frames[j][2]
            hist.append((b['wx'], b['wy'], b['wyaw']))
        hist = np.asarray(hist)
        v_now = np.linalg.norm(hist[-1, :2] - hist[-3, :2]) / 0.2
        v_prev = np.linalg.norm(hist[4, :2] - hist[2, :2]) / 0.2
        yaw_rate = float(np.arctan2(np.sin(hist[-1, 2] - hist[-5, 2]),
                                    np.cos(hist[-1, 2] - hist[-5, 2]))) / 0.5
        ego_feat = np.concatenate([[v_now, v_prev, yaw_rate],
                                   _to_ego(hist[:, :2], p0, h0).reshape(-1)]).astype(np.float32)
        # 周车
        ag = np.zeros((N_MAX, 28), np.float32)
        am = np.zeros(N_MAX, np.float32)
        rows = []
        for b in frames[i][1]:
            if b['class'] == 'ego_car':
                continue
            q = np.array([b['wx'], b['wy']])
            rows.append((float(np.linalg.norm(q - p0)), b))
        rows.sort(key=lambda r: r[0])
        for k, (_, b) in enumerate(rows[:N_MAX]):
            q = np.array([b['wx'], b['wy']])
            rel = _to_ego(q[None], p0, h0)[0]
            dh = b['wyaw'] - h0
            ex = b['extent']
            # 10 帧历史：相对该车当前位置、在 ego-now 系下（与 Waymo 口径一致）
            hist = np.zeros((HIST, 2), np.float32)
            aid = b.get('id')
            if aid is not None and aid in id2pos:
                pts = []
                for j in range(i - HIST + 1, i + 1):
                    pj = id2pos[aid].get(j)
                    pts.append(np.nan if pj is None else pj)
                if all(p is not np.nan for p in pts) and not any(
                        isinstance(p, float) for p in pts):
                    P = np.asarray(pts, np.float64)
                    pe = _to_ego(P, p0, h0) - rel[None]
                    hist = pe.astype(np.float32)
            ag[k] = np.r_[[rel[0], rel[1], np.cos(dh), np.sin(dh),
                           float(b.get('speed') or 0.0), 2 * ex[0], 2 * ex[1], 1.0],
                          hist.reshape(-1)]
            am[k] = 1.0
        # 地图
        mid = 0.5 * (A + B)
        dist = np.linalg.norm(mid - p0[None], axis=1)
        order = np.argsort(dist)[:M_MAX]
        mp = np.zeros((M_MAX, 7), np.float32)
        mm = np.zeros(M_MAX, np.float32)
        for k, idx in enumerate(order):
            if dist[idx] > MAP_R:
                break
            m = _to_ego(mid[idx][None], p0, h0)[0]
            ln = float(np.linalg.norm(B[idx] - A[idx]))
            hh = float(np.arctan2(B[idx][1] - A[idx][1], B[idx][0] - A[idx][0])) - h0
            mp[k] = [m[0], m[1], ln, np.cos(hh), np.sin(hh), 1.0, 1.0]
            mm[k] = 1.0
        # GT
        gt = np.stack([_to_ego(np.array([[frames[i + t * STEP][2]['wx'],
                                          frames[i + t * STEP][2]['wy']]]), p0, h0)[0]
                       for t in range(1, T_FUT + 1)]).astype(np.float32)
        out.append(dict(ego=ego_feat, agent=ag, amask=am, map=mp, mmask=mm, gt=gt,
                        fid=int(frames[i][0])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--route', required=True)
    ap.add_argument('--map', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    samp = build(args.route, args.map)
    if not samp:
        print('失败')
        return
    print(f'样本数 {len(samp)}')
    s = samp[0]
    for k, v in s.items():
        print(f'  {k}: {v.shape}')
    print(f'  gt 前3: {np.round(s["gt"][:3], 2).tolist()}')
    print(f'  周车数 {int(s["amask"].sum())}, 地图段 {int(s["mmask"].sum())}')
    np.savez_compressed(args.out, **{k: np.stack([d[k] for d in samp])
                                     for k in samp[0]})
    print('saved', args.out)


if __name__ == '__main__':
    main()
