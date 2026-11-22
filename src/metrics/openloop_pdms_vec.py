"""向量化 PDMS：与 openloop_pdms.score_trajectory 逐项对齐，一次算一批候选。

用法（训练里）：
    g = geo_from_npz(z, i)                 # 自车系几何
    out = score_batch(g, trajs)            # trajs (M,40,2) 自车系
    out['driving_score']                   # (M,)

对齐口径：
    - 数学与 openloop_pdms.step_metrics/score_trajectory 完全一致；
    - geo_from_scene() 用评测同一份 Scene（全量周车 + 全量地图）→ 与
      score_trajectory 应逐位相同（<1e-4）；
    - geo_from_npz() 用数据集里存的紧凑几何（最近 N_MAX 个周车 +
      沿专家走廊 R_MAX 条线段 + 构建期算好的 hm_*）→ 近似，差异会被量化。
"""
import numpy as np

from .openloop_pdms import (EPS, OFFROAD_MARGIN, _obb_overlap,  # noqa
                            ttc_score)

T_FUT = 40


# ---------------- 几何容器 ----------------
class Geo:
    """一帧场景的评分几何（某个坐标系下的世界表示）。"""

    __slots__ = ('ap', 'av', 'ah', 'ad', 'ea', 'eb', 'la', 'lb',
                 'expert_fut', 'expert_head', 'hm', 'ego_dim', 'frame',
                 'ap_f', 'av_f', 'ah_f')

    def __init__(self, ap, av, ah, ad, ea, eb, la, lb,
                 expert_fut, expert_head, hm, ego_dim, frame='world'):
        self.ap = np.ascontiguousarray(ap, np.float64)
        self.av = np.ascontiguousarray(av, bool)
        self.ah = np.ascontiguousarray(ah, np.float64)
        self.ad = np.ascontiguousarray(ad, np.float64)
        # 帧优先视图：(T,A,2) / (T,A)，避免广播错位
        self.ap_f = np.ascontiguousarray(self.ap.transpose(1, 0, 2))
        self.av_f = np.ascontiguousarray(self.av.T)
        self.ah_f = np.ascontiguousarray(self.ah.T)
        self.ea = np.ascontiguousarray(ea, np.float64)
        self.eb = np.ascontiguousarray(eb, np.float64)
        self.la = np.ascontiguousarray(la, np.float64)
        self.lb = np.ascontiguousarray(lb, np.float64)
        self.expert_fut = np.ascontiguousarray(expert_fut, np.float64)
        self.expert_head = np.ascontiguousarray(expert_head, np.float64)
        self.hm = hm
        self.ego_dim = tuple(float(x) for x in ego_dim)
        self.frame = frame

    @property
    def n_future(self):
        return len(self.hm['coll'])


def geo_from_scene(sc):
    """精确几何：直接取评测的 Scene（周车全量、地图全量）。"""
    T = sc.n_future
    hm = sc.human
    hm = {k: np.asarray(v, np.float64) for k, v in hm.items() if k != 'traj'}
    return Geo(sc.ap[:, sc.f0:sc.f0 + T], sc.av[:, sc.f0:sc.f0 + T],
               sc.ah[:, sc.f0:sc.f0 + T], sc.ad,
               sc.ea, sc.eb, sc.la, sc.lb,
               sc.expert_future, sc.expert_fut_heading, hm, sc.ego_dim,
               frame='world')


def _rot_to_ego(pts, p0, h0):
    d = np.asarray(pts, np.float64) - np.asarray(p0, np.float64)
    c, s = np.cos(-h0), np.sin(-h0)
    return np.stack([d[..., 0] * c - d[..., 1] * s,
                     d[..., 0] * s + d[..., 1] * c], -1)


def _has(z, key):
    files = getattr(z, 'files', None)
    return (key in files) if files is not None else (key in z)


def geo_from_npz(z, i):
    """紧凑几何（自车系）：数据集里存的周车未来 + 走廊线段 + hm_*。"""
    x, y, h0 = (float(v) for v in z['ego_world'][i])
    n = int(np.sum(z['amask'][i] > 0.5))
    ap = _rot_to_ego(z['agent_fut'][i, :n], (x, y), h0)
    av = z['agent_fut_valid'][i, :n] > 0.5
    ah = np.asarray(z['agent_fut_head'][i, :n], np.float64) - h0
    ad = np.stack([z['agent'][i, :n, 5], z['agent'][i, :n, 6]], 1).astype(np.float64)

    m = int(z['n_map_r'][i]) if _has(z, 'n_map_r') else 0
    seg = np.asarray(z['map_r'][i, :m], np.float64)
    is_lane = seg[:, 4] > 0.5
    ea, eb = seg[~is_lane][:, :2], seg[~is_lane][:, 2:4]
    la, lb = seg[is_lane][:, :2], seg[is_lane][:, 2:4]

    expert_fut = _rot_to_ego(z['expert40'][i], (x, y), h0)
    expert_head = np.asarray(z['expert_head40'][i], np.float64) - h0
    hm = {}
    for k in ('ttc', 'jerk', 'lat'):
        hm[k] = np.asarray(z['hm_' + k][i], np.float64)
    for k in ('coll', 'off', 'ddc'):
        hm[k] = np.asarray(z['hm_' + k][i]) > 0.5
    return Geo(ap, av, ah, ad, ea, eb, la, lb, expert_fut, expert_head, hm,
               z['ego_dim'][i], frame='ego')


# ---------------- 距离 ----------------
def seg_min_dist(P, A, B, chunk=1024):
    """P (M,2) 到线段集 (A,B) 的最小距离，返回 (M,)。"""
    P = np.asarray(P, np.float64)
    A = np.asarray(A, np.float64)
    B = np.asarray(B, np.float64)
    if len(A) == 0:
        return np.full(len(P), 1e9)
    AB = B - A
    den = np.maximum((AB ** 2).sum(1), 1e-12)
    out = np.empty(len(P))
    for i0 in range(0, len(P), chunk):
        p = P[i0:i0 + chunk]
        diff = p[:, None, :] - A[None]          # (m,S,2)
        t = np.clip((diff * AB[None]).sum(2) / den[None], 0.0, 1.0)
        proj = A[None] + t[..., None] * AB[None]
        out[i0:i0 + chunk] = np.linalg.norm(p[:, None, :] - proj, axis=2).min(1)
    return out


def _progress(g, q):
    """q (M,2) 投影到未来专家轨迹的进度 0~1（与 Scene.progress 一致）。"""
    A_, B_ = g.expert_fut[:-1], g.expert_fut[1:]
    if len(A_) == 0:
        return np.zeros(len(q))
    AB = B_ - A_
    den = np.maximum((AB ** 2).sum(1), 1e-12)
    L = np.linalg.norm(AB, axis=1)
    diff = q[:, None, :] - A_[None]
    t = np.clip((diff * AB[None]).sum(2) / den[None], 0.0, 1.0)
    proj = A_[None] + t[..., None] * AB[None]
    j = np.argmin(np.linalg.norm(q[:, None, :] - proj, axis=2), axis=1)
    cum = np.cumsum(L)
    arc = cum[j] - L[j] + t[np.arange(len(q)), j] * L[j]
    return arc / max(float(L.sum()), 1e-6)


# ---------------- 各项指标 ----------------
def _heading_speed(traj, dt=0.1):
    G, T = traj.shape[:2]
    d = np.diff(traj, axis=1) if T > 1 else np.zeros((G, 1, 2))
    speed = np.zeros((G, T))
    speed[:, :T - 1] = np.linalg.norm(d, axis=2) / dt
    heading = np.zeros((G, T))
    heading[:, :T - 1] = np.arctan2(d[..., 1], d[..., 0])
    if T > 1:
        heading[:, -1] = heading[:, -2]
    return heading, speed


def _collisions(g, traj, heading):
    G, T = traj.shape[:2]
    coll = np.zeros((G, T), bool)
    if g.ap.shape[0] == 0:
        return coll
    dist = np.linalg.norm(traj[:, :, None, :] - g.ap_f[None], axis=3)   # (G,T,A)
    hit = (dist < 6.0) & g.av_f[None]
    gi, ti, ai = np.nonzero(hit)
    for x, y, z in zip(gi, ti, ai):
        if coll[x, y]:
            continue
        if _obb_overlap(traj[x, y], float(heading[x, y]), g.ego_dim,
                        g.ap_f[y, z], float(g.ah_f[y, z]), g.ad[z]):
            coll[x, y] = True
    return coll


def _ttc(g, traj, heading, speed):
    G, T = traj.shape[:2]
    if g.ap.shape[0] == 0:
        return np.full((G, T), np.inf)
    dv = g.ap_f[None] - traj[:, :, None, :]                            # (G,T,A,2)
    dist = np.linalg.norm(dv, axis=3)
    ch, sh = np.cos(heading), np.sin(heading)
    fwd = dv[..., 0] * ch[..., None] + dv[..., 1] * sh[..., None]
    ok = (g.av_f[None]
          & (dist > 1e-3) & (dist <= 60.0)
          & (fwd >= 0.5 * dist)
          & (speed[:, :, None] > 1e-6))
    ttc = np.full(dist.shape, np.inf)
    val = dist / np.maximum(speed[:, :, None], 1e-12)
    ttc[ok] = val[ok]
    return ttc.min(2)


def _ratio(bad, good, T):
    """与 score_trajectory 内的 ratio 一致：good 为 0/inf 处记 1。"""
    good = np.asarray(good, np.float64)[None]
    out = np.ones_like(np.asarray(bad, np.float64))
    valid = np.isfinite(good) & (good > EPS)
    r = np.clip(good / np.maximum(bad, EPS), 0.0, 1.0)
    return np.where(valid, r, out)


# ---------------- 主入口 ----------------
def score_batch(g, trajs, dt=0.1, want_detail=True):
    """trajs (M,T,2) → dict of (M,) 指标。"""
    trajs = np.asarray(trajs, np.float64)
    if trajs.ndim == 2:
        trajs = trajs[None]
    M, T = trajs.shape[:2]
    T = int(min(T, g.n_future))
    trajs = trajs[:, :T]
    hm = g.hm
    heading, speed = _heading_speed(trajs, dt)

    coll = _collisions(g, trajs, heading)
    ttc = _ttc(g, trajs, heading, speed)
    P = trajs.reshape(-1, 2)
    jerk = np.zeros((M, T))
    if T > 1:
        jerk[:, 1:] = np.abs(np.diff(speed, axis=1)) / dt
    ddc = np.abs(np.arctan2(np.sin(heading - g.expert_head[:T][None]),
                            np.cos(heading - g.expert_head[:T][None]))) > np.pi / 2
    off = seg_min_dist(P, g.ea, g.eb) < OFFROAD_MARGIN
    off = off.reshape(M, T)
    lat = (seg_min_dist(P, g.la, g.lb).reshape(M, T) if len(g.la) > 0
           else np.zeros((M, T)))

    hm_coll = np.asarray(hm['coll'], bool)[:T]
    hm_off = np.asarray(hm['off'], bool)[:T]
    hm_ddc = np.asarray(hm['ddc'], bool)[:T]
    nc = np.where(np.any(coll & ~hm_coll[None], axis=1), 0.0, 1.0)
    dac = np.where(np.any(off & ~hm_off[None], axis=1), 0.0, 1.0)
    ddc_g = np.where(np.any(ddc & ~hm_ddc[None], axis=1), 0.0, 1.0)

    ttc_s = ttc_score(ttc.reshape(-1), np.tile(hm['ttc'][:T].astype(np.float64), M))
    ttc_s = ttc_s.reshape(M, T)
    comfort = _ratio(jerk, hm['jerk'][:T], T)
    comfort = np.where((np.asarray(hm['jerk'][:T])[None] <= EPS) & (jerk <= EPS), 1.0, comfort)
    comfort = np.where((np.asarray(hm['jerk'][:T])[None] <= EPS) & (jerk > EPS), 0.0, comfort)
    lk = _ratio(lat, hm['lat'][:T], T)
    lk = np.where((np.asarray(hm['lat'][:T])[None] <= EPS) & (lat <= EPS), 1.0, lk)
    lk = np.where((np.asarray(hm['lat'][:T])[None] <= EPS) & (lat > EPS), 0.0, lk)

    ep = np.minimum(1.0, _progress(g, trajs[:, -1]))
    ttc_avg = ttc_s.mean(1)
    comfort_avg = comfort.mean(1)
    lk_avg = lk.mean(1)
    weighted = (5 * ep + 5 * ttc_avg + 2 * comfort_avg + 2 * lk_avg) / 14.0
    score = nc * dac * ddc_g * weighted * 100.0
    out = {'driving_score': score, 'nc': nc, 'dac': dac, 'ddc': ddc_g,
           'ep': ep, 'ttc': ttc_avg, 'comfort': comfort_avg, 'lk': lk_avg}
    if want_detail:
        out['coll'] = coll
        out['off'] = off
        out['raw_ttc'] = ttc
        out['lat'] = lat
    return out


def score_one(g, traj, dt=0.1):
    o = score_batch(g, np.asarray(traj, np.float64)[None], dt=dt)
    return {k: (float(v[0]) if isinstance(v, np.ndarray) and v.ndim == 1 else v)
            for k, v in o.items()}
