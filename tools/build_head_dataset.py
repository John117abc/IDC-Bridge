"""构建规划头的训练数据：固定锚点（K-Means）+ 场景特征 + 标签。

锚点：对自车系（frame 10 为原点、朝向 0）的 80 帧 GT 未来做 K-Means → K 条固定锚点
特征：
    ego   : (3,)   当前速度 / 5 帧前的速度 / 偏航角速度
    agent : (N,8)  自车系相对位置(2) + 相对朝向(cos,sin) + 速度 + 长宽(2) + valid
    map   : (M,7)  自车系线段中点(2) + 长度 + 朝向(cos,sin) + 类型(lane/edge)
标签：
    label (K,)  最近锚点（按 FDE）
    resid (80,2) GT - 锚点（自车系）

用法：
    python tools/build_head_dataset.py --data-dir /data/data_json/training \
        --n-scenes 1500 --out /workspace/data/head --k 32
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

N_MAX = 32
M_MAX = 128
T_FUT = 80
HIST = 10


def to_ego(P, p0, h0):
    d = P - p0
    c, s = np.cos(-h0), np.sin(-h0)
    return np.stack([d[..., 0] * c - d[..., 1] * s,
                     d[..., 0] * s + d[..., 1] * c], -1)


def kmeans(X, k, iters=25, seed=0):
    rng = np.random.default_rng(seed)
    C = np.empty((k, X.shape[1]), np.float32)
    C[0] = X[rng.integers(len(X))]
    d2 = ((X - C[0]) ** 2).sum(1)
    for i in range(1, k):
        p = d2 / max(d2.sum(), 1e-9)
        C[i] = X[rng.choice(len(X), p=p)]
        d2 = np.minimum(d2, ((X - C[i]) ** 2).sum(1))
    for _ in range(iters):
        d2 = ((X[:, None, :] - C[None]) ** 2).sum(-1)
        lab = d2.argmin(1)
        for i in range(k):
            m = lab == i
            if m.any():
                C[i] = X[m].mean(0)
    return C, lab


_G = {}


def _gt_only(fp):
    """pass1 用：只取自车系 GT 未来（不构造全部特征，省一半时间）。"""
    try:
        d = json.load(open(fp))
    except Exception:
        return None
    objs = d.get('objects') or []
    idx = (d.get('metadata') or {}).get('sdc_track_index')
    if idx is None:
        return None
    pos = objs[int(idx)].get('position') or []
    if len(pos) < HIST + T_FUT + 1:
        return None
    P = np.asarray([[q['x'], q['y']] for q in pos], np.float64)
    H = np.asarray(objs[int(idx)].get('heading') or [0.0] * len(pos), np.float64)
    t = HIST
    gt = to_ego(P[t + 1:t + 1 + T_FUT], P[t], float(H[t]))
    if len(gt) < T_FUT or np.abs(gt).max() > 500:
        return None
    return gt.astype(np.float32)


def _init_worker(anchors):
    _G['a'] = anchors


def _one(fp):
    try:
        d = json.load(open(fp))
    except Exception:
        return None
    return build_one(d, _G['a'])


def build_one(d, anchors):
    """返回一个样本的 numpy 特征与标签；不可用返回 None。"""
    objs = d.get('objects') or []
    meta = d.get('metadata') or {}
    idx = meta.get('sdc_track_index')
    if idx is None:
        return None
    sdc = objs[int(idx)]
    pos = sdc.get('position') or []
    if len(pos) < HIST + T_FUT + 1:
        return None
    P = np.asarray([[q['x'], q['y']] for q in pos], np.float64)
    H = np.asarray(sdc.get('heading') or [0.0] * len(pos), np.float64)
    t = HIST
    p0, h0 = P[t], float(H[t])
    if not np.all(np.isfinite(p0)):
        return None

    # --- ego 特征 ---
    v_now = np.linalg.norm(P[t + 1] - P[t]) / 0.1 if t + 1 < len(P) else 0.0
    j = max(t - 5, 0)
    v_prev = np.linalg.norm(P[j + 1] - P[j]) / 0.1 if j + 1 < len(P) else v_now
    yaw = float(np.arctan2(np.sin(H[t] - H[j]), np.cos(H[t] - H[j]))) / max(0.1 * (t - j), 1e-3)
    ego_hist = np.zeros((HIST, 2), np.float32)
    j0 = t - (HIST - 1)
    if j0 >= 0:
        ego_hist = (to_ego(P[j0:t + 1], p0, h0) - np.array([0.0, 0.0])).astype(np.float32)
    ego = np.concatenate([[v_now, v_prev, yaw], ego_hist.reshape(-1)]).astype(np.float32)

    # --- GT 未来（自车系）---
    fut = P[t + 1:t + 1 + T_FUT]
    if len(fut) < T_FUT:
        return None
    gt = to_ego(fut, p0, h0).astype(np.float32)
    if np.abs(gt).max() > 500:
        return None

    # --- agents ---
    ag = np.zeros((N_MAX, 28), np.float32)
    amask = np.zeros(N_MAX, np.float32)
    cands = []
    for i, o in enumerate(objs):
        if i == int(idx) or not o.get('position'):
            continue
        q = o['position']
        if len(q) <= t:
            continue
        p = np.array([q[t]['x'], q[t]['y']], np.float64)
        if not np.all(np.isfinite(p)):
            continue
        cands.append((np.linalg.norm(p - p0), i, o, p))
    cands.sort(key=lambda x: x[0])
    for n, (dist, i, o, p) in enumerate(cands[:N_MAX]):
        hh = float((o.get('heading') or [0.0])[t]) if o.get('heading') else 0.0
        dh = hh - h0
        q = o['position']
        v = 0.0
        if t + 1 < len(q):
            v = float(np.linalg.norm(np.array([q[t + 1]['x'], q[t + 1]['y']]) - p) / 0.1)
        rel = to_ego(p[None], p0, h0)[0]
        hist = np.zeros(20, np.float32)
        j0 = t - (HIST - 1)
        if j0 >= 0 and len(q) >= t + 1:
            pts = np.asarray([[q[k]['x'], q[k]['y']] for k in range(j0, t + 1)],
                             np.float64)
            pe = to_ego(pts, p0, h0) - rel[None]
            hist = pe.reshape(-1).astype(np.float32)
        ag[n] = np.r_[[rel[0], rel[1], np.cos(dh), np.sin(dh), v,
                       float(o.get('length') or 4.5),
                       float(o.get('width') or 1.9), 1.0], hist]
        amask[n] = 1.0

    # --- map（只取 30 m 内）---
    mp = np.zeros((M_MAX, 7), np.float32)
    mmask = np.zeros(M_MAX, np.float32)
    segs = []
    for r in (d.get('roads') or []):
        typ = r.get('type')
        if typ not in ('lane', 'road_line', 'road_edge') or len(r.get('geometry') or []) < 2:
            continue
        L = np.asarray([[q['x'], q['y']] for q in r['geometry']], np.float64)
        for a, b in zip(L[:-1], L[1:]):
            mid = 0.5 * (a + b)
            if np.linalg.norm(mid - p0) > 30:
                continue
            segs.append((np.linalg.norm(mid - p0), a, b, 1.0 if typ != 'road_edge' else 0.0))
    segs.sort(key=lambda x: x[0])
    for n, (dist, a, b, is_lane) in enumerate(segs[:M_MAX]):
        mid = to_ego((0.5 * (a + b))[None], p0, h0)[0]
        ln = float(np.linalg.norm(b - a))
        hh = float(np.arctan2(b[1] - a[1], b[0] - a[0])) - h0
        mp[n] = [mid[0], mid[1], ln, np.cos(hh), np.sin(hh), is_lane, 1.0]
        mmask[n] = 1.0

    # --- 标签 ---
    fde = np.linalg.norm(anchors[:, -1, :] - gt[-1][None], axis=1)
    k = int(np.argmin(fde))
    resid = gt - anchors[k]
    return dict(ego=ego, agent=ag, amask=amask, map=mp, mmask=mmask,
                gt=gt, label=np.int64(k), resid=resid.astype(np.float32),
                fde=float(fde[k]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--n-scenes", type=int, default=1500)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.data_dir, 'tfrecord-*.json')))
    rng = np.random.default_rng(args.seed)
    if len(files) > args.n_scenes:
        files = [files[i] for i in rng.choice(len(files), args.n_scenes,
                                              replace=False)]
    print(f"场景 {len(files)}")

    # ---- pass 1：收集自车系 GT 未来，聚类出锚点 ----
    from multiprocessing import Pool
    workers = min(8, os.cpu_count() or 4)
    with Pool(workers) as p:
        res = list(p.imap(_gt_only, files, chunksize=20))
    gts = [r for r in res if r is not None]
    keep = [fp for fp, r in zip(files, res) if r is not None]
    print(f"有效 {len(gts)}")
    X = np.asarray(gts, np.float32).reshape(len(gts), -1)
    C, _ = kmeans(X, args.k, seed=args.seed)
    anchors = C.reshape(args.k, T_FUT, 2)
    np.save(os.path.join(args.out, 'anchors.npy'), anchors)
    print(f"锚点 {anchors.shape}")

    # ---- pass 2：特征 + 标签 ----
    data = {k: [] for k in ('ego', 'agent', 'amask', 'map', 'mmask',
                            'gt', 'label', 'resid')}
    fdes = []
    with Pool(workers, initializer=_init_worker, initargs=(anchors,)) as p:
        for s in p.imap(_one, keep, chunksize=20):
            if s is None:
                continue
            for k in data:
                data[k].append(s[k])
            fdes.append(s['fde'])
    out = {k: np.asarray(v) for k, v in data.items()}
    np.savez_compressed(os.path.join(args.out, 'train.npz'), **out)
    print(f"样本 {len(fdes)}")
    f = np.asarray(fdes)
    print(f"标签 FDE: p50={np.median(f):.2f} p90={np.percentile(f,90):.2f} "
          f"max={f.max():.2f}  (<1m 占比 {100*np.mean(f<1):.0f}%)")
    print(f"标签分布(前 8): {np.bincount(out['label'], minlength=args.k)[:8]}")


if __name__ == "__main__":
    main()
