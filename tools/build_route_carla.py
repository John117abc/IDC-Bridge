"""route: 用车道路中心线（map npz）沿 GT 做匹配，生成 40 m 的 ego 系 route。

输出：route (N,4) = [x, y, s, heading_delta]，并画 2x3 对照图。
"""
import argparse
import glob
import gzip
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from matplotlib.patches import Polygon

T_NOW, HOR, STEP = 10, 40.0, 2.0


def _pose(b):
    m = np.asarray(b['matrix'], np.float64)
    return float(m[0][3]), float(m[1][3]), float(np.arctan2(m[1][0], m[0][0]))


def _load(fp):
    with gzip.open(fp, 'rt') as fh:
        return json.load(fh)


def ego_track(route_dir):
    fs = sorted(glob.glob(os.path.join(route_dir, 'boxes', '*.json.gz')))
    P = []
    for f in fs:
        e = [b for b in _load(f) if b['class'] == 'ego_car']
        if e:
            P.append(_pose(e[0]))
    return np.asarray(P)


def build_route(trk, lanes_npz, t=T_NOW, hor=HOR, step=STEP, tol=3.5):
    z = np.load(lanes_npz)
    L, NP, ID = z['lanes'], z['lane_pts'], z['lane_ids']
    polys, o = [], 0
    for k, n in enumerate(NP):
        polys.append((ID[k], L[o:o + n, :2]))
        o += n
    gt = trk[t:min(len(trk), t + 90)]
    if len(gt) < 5:
        return None
    hd = gt[min(4, len(gt) - 1)] - gt[0]
    hd = hd / max(np.linalg.norm(hd), 1e-9)
    allpts = np.concatenate([pl for _, pl in polys])
    owner = np.concatenate([np.full(len(pl), k, np.int32)
                            for k, (_, pl) in enumerate(polys)])
    tree = cKDTree(allpts)
    seq = []
    for p in gt:
        d, idx = tree.query(p[:2], k=1)
        if d < tol:
            k = int(owner[idx])
            if not seq or seq[-1] != k:
                seq.append(k)
    if not seq:
        return None
    hd0 = trk[min(t + 5, len(trk) - 1)][:2] - trk[t][:2]
    hd0 = hd0 / max(np.linalg.norm(hd0), 1e-9)
    pts = []
    for k in seq:
        pl = polys[k][1].copy()
        if not pts:
            if len(pl) >= 2 and float((pl[1] - pl[0]) @ hd0) < 0:
                pl = pl[::-1]          # 第一条车道按自车朝向定向
        else:
            if np.linalg.norm(pl[-1] - pts[-1]) < np.linalg.norm(pl[0] - pts[-1]):
                pl = pl[::-1]
        pts.extend(pl.tolist() if not pts else pl[1:].tolist())
    R = np.asarray(pts)
    dist = np.linalg.norm(R - trk[t][:2][None], axis=1)
    j = int(np.argmin(dist))
    R = R[j:]
    if len(R) >= 2:
        u = R[1] - R[0]
        ang0 = float(np.degrees(np.arccos(np.clip((u / max(np.linalg.norm(u), 1e-9)) @ hd0, -1, 1))))
        print('first_seg_angle = %.1f deg' % ang0)
    seg = np.linalg.norm(np.diff(R, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    R, s = R[s <= hor], s[s <= hor]
    if len(R) < 3:
        return None
    ss = np.arange(0.0, s[-1], step)
    R = np.stack([np.interp(ss, s, R[:, 0]), np.interp(ss, s, R[:, 1])], 1)
    ss = np.arange(0.0, len(R) * step, step)[:len(R)]
    return R, ss, hd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--route-dir', required=True)
    ap.add_argument('--map', required=True)
    ap.add_argument('--out', default='/mnt/2T_HDD/plant2_vector/route_view.png')
    args = ap.parse_args()
    trk = ego_track(args.route_dir)
    res = build_route(trk, args.map)
    if res is None:
        print('route 构建失败')
        return
    R, ss, hd = res
    p0, h0 = trk[T_NOW][0], trk[T_NOW][2]
    print(f'route 点数={len(R)} 长度={ss[-1]:.1f} m')

    z = np.load(args.map)
    L, NP = z['lanes'], z['lane_pts']
    fig, ax = plt.subplots(figsize=(8, 7))
    o = 0
    for n in NP:
        pl = L[o:o + n, :2]
        o += n
        m = np.linalg.norm(pl - trk[T_NOW][:2][None], axis=1) < 60
        if m.sum() > 1:
            ax.plot(pl[m, 0], pl[m, 1], '-', lw=0.7, c='#b0bec5', zorder=1)
    g = trk[T_NOW:T_NOW + 90]
    ax.plot(g[:, 0], g[:, 1], '--', lw=1.4, c='#000000', label='GT (label)', zorder=3)
    ax.plot(R[:, 0], R[:, 1], '-', lw=3.0, c='#d50000', label='route (lane centers)', zorder=5)
    ax.plot(p0, trk[T_NOW][1], 'o', ms=9, c='#2962ff', label='ego', zorder=6)
    ax.set_xlim(p0 - 45, p0 + 45)
    ax.set_ylim(trk[T_NOW][1] - 45, trk[T_NOW][1] + 45)
    ax.set_aspect('equal')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    ax.set_title(os.path.basename(args.route_dir)[:40], fontsize=9)
    plt.tight_layout()
    plt.savefig(args.out, dpi=110)
    print('saved', args.out)


if __name__ == '__main__':
    main()
