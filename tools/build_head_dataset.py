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
T_FUT = 8
T_FUT_EXT = 40      # 奖励用的 4 s 逐帧（0.1 s）          # 8 个 waypoint（对齐 DiffusionDrive）
WP_STEP = 5        # 每 0.5 s 一个（10 Hz × 5）
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
    widx = [t + WP_STEP * (i + 1) for i in range(T_FUT)]
    if widx[-1] >= len(P):
        return None
    gt = to_ego(P[widx], P[t], float(H[t]))
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


def build_one(d, anchors, t_now=None):
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
    t = int(t_now) if t_now is not None else HIST
    p0, h0 = P[t], float(H[t])
    if not np.all(np.isfinite(p0)):
        return None

    # --- ego 特征 ---
    # 因果速度：只用 [t] 与 [t-2]（不读 t+1，避免 1 帧泄漏）
    _k = 2 if t - 2 >= 0 else 1
    v_now = float(np.linalg.norm(P[t] - P[t - _k]) / (_k * 0.1))
    j = max(t - 5, 0)
    _k2 = 2 if j - 2 >= 0 else 1
    v_prev = float(np.linalg.norm(P[j] - P[j - _k2]) / (_k2 * 0.1))
    yaw = float(np.arctan2(np.sin(H[t] - H[j]), np.cos(H[t] - H[j]))) / max(0.1 * (t - j), 1e-3)
    ego_hist = np.zeros((HIST, 2), np.float32)
    j0 = t - (HIST - 1)
    if j0 >= 0:
        ego_hist = (to_ego(P[j0:t + 1], p0, h0) - np.array([0.0, 0.0])).astype(np.float32)
    ego = np.concatenate([[v_now, v_prev, yaw], ego_hist.reshape(-1)]).astype(np.float32)

    # --- GT 未来（自车系）---
    widx = [t + WP_STEP * (i + 1) for i in range(T_FUT)]
    if widx[-1] >= len(P):
        return None
    fut = P[widx]
    if len(fut) < T_FUT:
        return None
    gt = to_ego(fut, p0, h0).astype(np.float32)
    if np.abs(gt).max() > 500:
        return None

    # --- agents ---
    ag = np.zeros((N_MAX, 28), np.float32)
    amask = np.zeros(N_MAX, np.float32)
    agent_fut = np.zeros((N_MAX, T_FUT_EXT, 2), np.float32)
    agent_fut_valid = np.zeros((N_MAX, T_FUT_EXT), np.float32)
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
        # 因果速度：只用 [t] 与 [t-2]，不读 q[t+1]
        _ka = 2 if t - 2 >= 0 else 1
        if t - _ka >= 0:
            _qk = q[t - _ka]
            v = float(np.linalg.norm(p - np.array([_qk['x'], _qk['y']]))
                      / (_ka * 0.1))
        else:
            v = 0.0
        rel = to_ego(p[None], p0, h0)[0]
        hist = np.zeros(20, np.float32)
        j0 = t - (HIST - 1)
        if j0 >= 0 and len(q) >= t + 1:
            pts = np.asarray([[q[k]['x'], q[k]['y']] for k in range(j0, t + 1)],
                             np.float64)
            pe = to_ego(pts, p0, h0) - rel[None]
            hist = pe.reshape(-1).astype(np.float32)
        # 逐帧未来位置（奖励用；只对回放周车，评测口径）
        _val = o.get('valid') or [1] * len(q)
        for _j in range(T_FUT_EXT):
            _tj = t + 1 + _j
            if _tj < len(q) and int(_val[_tj]) == 1:
                _qq = q[_tj]
                agent_fut[n, _j] = [_qq['x'], _qq['y']]
                agent_fut_valid[n, _j] = 1.0
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
                fde=float(fde[k]), agent_fut=agent_fut,
                agent_fut_valid=agent_fut_valid,
                ego_world=np.asarray([p0[0], p0[1], h0], np.float32))


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
    # split_before_cluster：先划分，锚点只在训练集上聚类
    rng2 = np.random.default_rng(args.seed + 1)
    perm = rng2.permutation(len(files))
    n_val = max(1, int(round(0.1 * len(files))))
    val_idx = set(perm[:n_val].tolist())
    is_val = np.array([i in val_idx for i in range(len(files))], bool)
    train_files = [f for f, v in zip(files, is_val) if not v]
    print(f"场景 {len(files)}  训练 {len(train_files)}  验证 {int(is_val.sum())}")

    # ---- pass 1：收集自车系 GT 未来，聚类出锚点 ----
    from multiprocessing import Pool
    workers = min(8, os.cpu_count() or 4)
    with Pool(workers) as p:
        res = list(p.imap(_gt_only, train_files, chunksize=20))
    gts = [r for r in res if r is not None]
    keep_train = [fp for fp, r in zip(train_files, res) if r is not None]
    print(f"有效 {len(gts)}")
    X = np.asarray(gts, np.float32).reshape(len(gts), -1)
    C, _ = kmeans(X, args.k, seed=args.seed)
    anchors = C.reshape(args.k, T_FUT, 2)
    np.save(os.path.join(args.out, 'anchors.npy'), anchors)
    print(f"锚点 {anchors.shape}")

    # ---- pass 2：特征 + 标签 ----
    keys = ('ego', 'agent', 'amask', 'map', 'mmask', 'gt', 'label', 'resid',
            'agent_fut', 'agent_fut_valid', 'ego_world')
    data_tr = {k: [] for k in keys}
    ids_tr = []
    data_va = {k: [] for k in keys}
    ids_va = []
    fdes = []
    _train_names = set(os.path.basename(x) for x in train_files)
    with Pool(workers, initializer=_init_worker, initargs=(anchors,)) as p:
        for fp, smp in zip(files, p.imap(_one, files, chunksize=20)):
            if smp is None:
                continue
            _v = os.path.basename(fp) in _train_names
            _dst, _ids = (data_tr, ids_tr) if _v else (data_va, ids_va)
            for k in keys:
                _dst[k].append(smp[k])
            _ids.append(os.path.basename(fp))
            fdes.append(smp['fde'])
    out = {k: np.asarray(v) for k, v in data_tr.items()}
    out['scene_id'] = np.asarray(ids_tr)
    np.savez_compressed(os.path.join(args.out, 'train.npz'), **out)
    if ids_va:
        outv = {k: np.asarray(v) for k, v in data_va.items()}
        outv['scene_id'] = np.asarray(ids_va)
        np.savez_compressed(os.path.join(args.out, 'val.npz'), **outv)
        print(f"验证集 {len(ids_va)} 条 → val.npz")
    print(f"样本 {len(fdes)}")
    f = np.asarray(fdes)
    print(f"标签 FDE: p50={np.median(f):.2f} p90={np.percentile(f,90):.2f} "
          f"max={f.max():.2f}  (<1m 占比 {100*np.mean(f<1):.0f}%)")
    print(f"标签分布(前 8): {np.bincount(out['label'], minlength=args.k)[:8]}")


if __name__ == "__main__":
    main()
