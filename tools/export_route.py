"""S1: route 生成（状态机 + 方向校验 + 接缝转角约束）。纯地图，不使用 GT 形状。"""
import argparse
import json
import os

import numpy as np

T_NOW = 10
HORIZON = 30.0
RESAMPLE = 3.0
START_TOL = 6.0
JOIN_TOL = 6.0
MAX_LANES = 12
COS45 = 0.707
COS90 = 0.0


def _lanes(d):
    out = []
    for r in (d.get('roads') or []):
        typ = r.get('type')
        if typ not in ('lane', 'road_line') or len(r.get('geometry') or []) < 2:
            continue
        L = np.asarray([[q['x'], q['y']] for q in r['geometry']], np.float64)
        if np.linalg.norm(np.diff(L, axis=0), axis=1).sum() < 3.0:
            continue
        out.append(L)
    return out


def _seg_dist(P, A, B):
    AB = B - A
    t = np.clip(np.sum((P - A) * AB, -1) / np.maximum(np.sum(AB * AB, -1), 1e-9), 0., 1.)
    return np.linalg.norm(P - (A + t[..., None] * AB), axis=-1), t


def _u(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else np.array([1.0, 0.0])


def _udir(L, i):
    return _u(L[min(i + 1, len(L) - 1)] - L[i])


def _gt_near(gt, p, t_now=T_NOW):
    k = int(np.argmin(np.linalg.norm(gt[t_now:] - p[None], axis=1))) + t_now
    return k


def _gt_dir(gt, k, t_now=T_NOW):
    a, b = gt[max(k - 2, t_now)], gt[min(k + 2, len(gt) - 1)]
    return _u(b - a)


def build_route(d, t_now=T_NOW, horizon=HORIZON, step=RESAMPLE):
    objs = d['objects']
    sdc = objs[int(d['metadata']['sdc_track_index'])]
    gt = np.asarray([[q['x'], q['y']] for q in sdc['position']], np.float64)
    p0 = gt[t_now]
    Ls = _lanes(d)
    if not Ls:
        return None, gt, None
    hd0 = _gt_dir(gt, t_now, t_now)

    # ---- 起点车道 ----
    best = None
    for li, L in enumerate(Ls):
        dist, t = _seg_dist(p0[None], L[:-1], L[1:])
        j = int(np.argmin(dist))
        if dist[j] > START_TOL:
            continue
        if float(_udir(L, j) @ hd0) < COS45:
            continue
        if best is None or dist[j] < best[0]:
            best = (float(dist[j]), li, j)
    if best is None:
        return None, gt, None
    _, li, j0 = best
    L = Ls[li]
    dist, t = _seg_dist(p0[None], L[j0][None], L[j0 + 1][None])
    pts = [L[j0] + float(t[0]) * (L[j0 + 1] - L[j0])]
    pts.extend(L[j0 + 1:].tolist())
    used = {li}
    seams = []

    # ---- 沿拓扑前进 ----
    for _ in range(MAX_LANES):
        R = np.asarray(pts)
        if float(np.linalg.norm(np.diff(R, axis=0), axis=1).sum()) >= horizon:
            break
        end = R[-1]
        hd_end = _u(R[-1] - R[-3]) if len(R) >= 3 else _u(R[-1] - R[0])
        cand = None
        for nli, NL in enumerate(Ls):
            if nli in used:
                continue
            d0, d1 = np.linalg.norm(NL[0] - end), np.linalg.norm(NL[-1] - end)
            P2 = NL[::-1] if d1 < d0 else NL
            dd = min(d0, d1)
            if dd > JOIN_TOL:
                continue
            u0 = _udir(P2, 0)
            u1 = _u(P2[min(5, len(P2) - 1)] - P2[0])
            if float(u0 @ hd_end) < COS45:          # 接缝转角 < 45°
                continue
            if float(u1 @ hd_end) < COS90:          # 不许接上后立刻掉头
                continue
            ang = float(np.degrees(np.arccos(np.clip(u0 @ hd_end, -1, 1))))
            cov = 0.0
            for q in gt[t_now:]:
                _dd, _tt = _seg_dist(q[None], P2[:-1], P2[1:])
                if _dd.min() < 4.0:
                    _j = int(np.argmin(_dd))
                    cov = max(cov, float(np.linalg.norm(P2[_j] - P2[0])))
            sc = cov - 2.0 * ang
            if cand is None or sc > cand[0]:
                cand = (sc, nli, P2, ang)
        if cand is None:
            break
        _, nli, P2, ang = cand
        used.add(nli)
        seams.append(ang)
        pts.extend(P2[1:].tolist())

    R = np.asarray(pts, np.float64)
    if len(R) < 3:
        return None, gt, None

    # ---- 方向校验：首段必须与 GT 同处方向一致，否则整条翻转 ----
    k = _gt_near(gt, R[0], t_now)
    gdir = _gt_dir(gt, k, t_now)
    a0 = float(np.degrees(np.arccos(np.clip(_u(R[1] - R[0]) @ gdir, -1, 1))))
    if a0 > 90.0:
        R = R[::-1]
        a0 = float(np.degrees(np.arccos(np.clip(_u(R[1] - R[0]) @ gdir, -1, 1))))
        seams = [180.0 - s for s in seams]

    # ---- 重采样 ----
    seg = np.linalg.norm(np.diff(R, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    keep = s <= horizon
    R, s = R[keep], s[keep]
    if len(R) < 3:
        return None, gt, None
    ss = np.arange(0.0, s[-1], step)
    out = np.stack([np.interp(ss, s, R[:, 0]), np.interp(ss, s, R[:, 1])], 1)
    info = dict(first_angle=a0, max_seam=max(seams) if seams else 0.0,
                n_seam=len(seams))
    return out, gt, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='/data/data_json/training')
    ap.add_argument('--names', default='')
    ap.add_argument('--out', default='/workspace/data/route_smoke')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for nm in [x for x in args.names.split(',') if x]:
        fp = os.path.join(args.data_dir, nm)
        if not os.path.exists(fp):
            print('missing', fp)
            continue
        route, gt, info = build_route(json.load(open(fp)))
        if route is None or len(route) < 2:
            print('%-38s 失败' % nm)
            continue
        Ln = float(np.linalg.norm(np.diff(route, axis=0), axis=1).sum())
        dev = [float(np.min(np.linalg.norm(gt[T_NOW:] - q[None], axis=1))) for q in route]
        ok = (info['first_angle'] < 30 and info['max_seam'] < 45
              and float(np.median(dev)) < 5)
        print('%-38s 长%4.0fm 首段%5.1f° 最大接缝%5.1f° 距GT中位%4.1f m  %s'
              % (nm, Ln, info['first_angle'], info['max_seam'],
                 float(np.median(dev)), 'OK' if ok else 'BAD'))
        np.save(os.path.join(args.out, nm.replace('.json', '_route.npy')), route)


if __name__ == '__main__':
    main()
