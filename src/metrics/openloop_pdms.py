"""开环 PDMS（相对 log、带历史/未来切分）。

Waymo 场景约定（已在 tfrecord 上验证）：
    91 帧 @10Hz，current_time_index = 10
    frames  0..9  = **历史 1 秒**
    frame     10  = 现在
    frames 11..90 = **未来 8 秒**  ← 预测/评测窗口

评分时：周车按日志在未来 8 秒走，自车按给定轨迹走；
硬指标按"人类也违反则不计"过滤，软指标相对人类归一 → 人类自评 = 100。
"""
import glob
import json
import math
import os

import numpy as np

DIM = {'vehicle': (4.8, 1.95), 'pedestrian': (0.9, 0.8), 'cyclist': (1.8, 0.7)}
EGO_DIM = (4.8, 1.95)
T_NOW = 10                 # 决策时刻（1.0 s）
OFFROAD_MARGIN = 1.0
TTC_HORIZON = 5.0
EPS = 1e-6


def _poly(arr):
    return np.asarray([[q['x'], q['y']] for q in arr], dtype=np.float64)


def _seg_arrays(polys):
    A, B = [], []
    for L in polys:
        if len(L) >= 2:
            A.append(L[:-1]); B.append(L[1:])
    if not A:
        return np.zeros((0, 2)), np.zeros((0, 2))
    return np.vstack(A), np.vstack(B)


def _min_dist(q, a, b):
    if len(a) == 0:
        return 1e9
    ab = b - a
    den = np.maximum((ab ** 2).sum(1), 1e-12)
    t = np.clip(((q - a) * ab).sum(1) / den, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return float(np.linalg.norm(q[None, :] - proj, axis=1).min())


def _obb_overlap(pa, ha, da, pb, hb, db):
    def corners(p, h, d):
        c, s = math.cos(h), math.sin(h)
        L, W = d[0] / 2, d[1] / 2
        pts = np.array([[-L, -W], [L, -W], [L, W], [-L, W]])
        return pts @ np.array([[c, -s], [s, c]]).T + np.asarray(p)
    A, B = corners(pa, ha, da), corners(pb, hb, db)
    for poly in (A, B):
        for k in range(4):
            e = poly[(k + 1) % 4] - poly[k]
            ax = np.array([-e[1], e[0]])
            n = np.linalg.norm(ax)
            if n < 1e-9:
                continue
            ax /= n
            pa_, pb_ = A @ ax, B @ ax
            if pa_.max() < pb_.min() or pb_.max() < pa_.min():
                return False
    return True


class Scene:
    def __init__(self, d):
        objs = d.get('objects') or []
        idx = (d.get('metadata') or {}).get('sdc_track_index')
        if idx is None:
            idx = next((i for i, o in enumerate(objs) if o.get('mark_as_expert')), 0)
        self.sdc_idx = int(idx)
        sdc = objs[self.sdc_idx]
        self.T = len(sdc.get('position') or [])
        self.t_now = min(T_NOW, max(self.T - 2, 0))
        self.f0 = self.t_now + 1                  # 未来起点
        self.f1 = self.T
        self.n_future = max(self.f1 - self.f0, 0)

        self.ego_dim = (float(sdc.get('length') or EGO_DIM[0]),
                        float(sdc.get('width') or EGO_DIM[1]))
        self.expert_pos = _poly(sdc['position'])
        self.expert_heading = np.asarray(sdc.get('heading') or [0.0] * self.T,
                                         dtype=np.float64)
        self.expert_future = self.expert_pos[self.f0:self.f1]
        self.expert_fut_heading = self.expert_heading[self.f0:self.f1]
        self.xa, self.xb = _seg_arrays([self.expert_future])

        ap, ah, av, ad = [], [], [], []
        for i, o in enumerate(objs):
            if i == self.sdc_idx or not o.get('position'):
                continue
            p = _poly(o['position'])
            if len(p) < self.T:
                continue
            ap.append(p)                                   # 保留全 91 帧（历史可用）
            ah.append(np.asarray(o.get('heading') or [0.0] * len(p),
                                 dtype=np.float64)[:len(p)])
            av.append(np.asarray(o.get('valid') or [True] * len(p),
                                 dtype=bool)[:len(p)])
            dflt = DIM.get(o.get('type'), DIM['vehicle'])
            ad.append((float(o.get('length') or dflt[0]),
                       float(o.get('width') or dflt[1])))
        if ap:
            self.ap, self.ah = np.stack(ap), np.stack(ah)
            self.av, self.ad = np.stack(av), np.asarray(ad, dtype=np.float64)
        else:
            self.ap = np.zeros((0, self.T, 2)); self.ah = np.zeros((0, self.T))
            self.av = np.zeros((0, self.T), dtype=bool); self.ad = np.zeros((0, 2))

        edge, lane = [], []
        for r in (d.get('roads') or []):
            if len(r.get('geometry') or []) < 2:
                continue
            L = _poly(r['geometry'])
            if r.get('type') == 'road_edge':
                edge.append(L)
            elif r.get('type') in ('lane', 'road_line'):
                lane.append(L)
        self.ea, self.eb = _seg_arrays(edge)
        self.la, self.lb = _seg_arrays(lane)
        self._human = None


    # ---- 决策时刻的便捷访问（历史/未来切分之后）----
    def pose_now(self):
        """决策时刻（frame t_now）的 (x, y, heading)。"""
        j = min(self.t_now, self.T - 1)
        p = self.expert_pos[j]
        if j + 1 < self.T:
            d = self.expert_pos[j + 1] - p
            h = float(np.arctan2(d[1], d[0])) if np.linalg.norm(d) > 0.05 \
                else float(self.expert_heading[j])
        else:
            h = float(self.expert_heading[j])
        return float(p[0]), float(p[1]), h

    def speed_now(self, dt=0.1):
        j = min(self.t_now, self.T - 2)
        return float(np.linalg.norm(self.expert_pos[j + 1] - self.expert_pos[j]) / dt)
    def progress(self, q):
        """q 投影到**未来专家轨迹**上的进度 0~1。"""
        if len(self.xa) == 0:
            return 0.0
        ab = self.xb - self.xa
        den = np.maximum((ab ** 2).sum(1), 1e-12)
        t = np.clip(((q - self.xa) * ab).sum(1) / den, 0.0, 1.0)
        proj = self.xa + t[:, None] * ab
        j = int(np.argmin(np.linalg.norm(q[None, :] - proj, axis=1)))
        L = np.linalg.norm(ab, axis=1)
        arc = float(L[:j].sum() + t[j] * L[j])
        return arc / max(float(L.sum()), 1e-6)

    @property
    def human(self):
        if self._human is None:
            self._human = step_metrics(self, self.expert_future)
        return self._human


def step_metrics(scene: Scene, traj, dt=0.1):
    """在**未来窗口**上逐时刻量指标。traj 长度应 = n_future。"""
    nf = scene.n_future
    T = min(len(traj), nf)
    traj = np.asarray(traj[:T], dtype=np.float64)
    d = np.diff(traj, axis=0) if T > 1 else np.zeros((1, 2))
    speed = np.r_[np.linalg.norm(d, axis=1) / dt, 0.0]
    heading = np.r_[np.arctan2(d[:, 1], d[:, 0]), 0.0]
    heading[-1] = heading[-2] if T > 1 else 0.0

    coll = np.zeros(T, dtype=bool)
    off = np.zeros(T, dtype=bool)
    ddc = np.zeros(T, dtype=bool)
    ttc = np.full(T, np.inf)
    jerk = np.zeros(T)
    lat = np.zeros(T)
    prev_v = None

    for t in range(T):
        at = scene.f0 + t                       # 绝对帧号
        p = traj[t]; h = float(heading[t]); v = float(speed[t])
        if len(scene.ap):
            dd = np.linalg.norm(scene.ap[:, at, :] - p[None, :], axis=1)
            cand = np.where((dd < 6.0) & scene.av[:, at])[0]
            for a in cand:
                if _obb_overlap(p, h, scene.ego_dim, scene.ap[a, at],
                                scene.ah[a, at], scene.ad[a]):
                    coll[t] = True
                    break
            best = np.inf
            for a in range(len(scene.ap)):
                if not scene.av[a, at]:
                    continue
                dv = scene.ap[a, at] - p
                dist = float(np.linalg.norm(dv))
                if dist < 1e-3 or dist > 60:
                    continue
                if float(dv[0] * math.cos(h) + dv[1] * math.sin(h)) < 0.5 * dist:
                    continue
                if v > 1e-6:
                    best = min(best, dist / v)
            ttc[t] = best
        off[t] = len(scene.ea) > 0 and _min_dist(p, scene.ea, scene.eb) < OFFROAD_MARGIN
        hf = scene.expert_fut_heading[t]
        ddc[t] = abs(math.atan2(math.sin(h - hf), math.cos(h - hf))) > math.pi / 2
        if prev_v is not None:
            jerk[t] = abs(v - prev_v) / dt
        prev_v = v
        lat[t] = _min_dist(p, scene.la, scene.lb) if len(scene.la) else 0.0
    return {'coll': coll, 'off': off, 'ddc': ddc, 'ttc': ttc,
            'jerk': jerk, 'lat': lat, 'traj': traj}


def score_trajectory(scene: Scene, traj, dt=0.1):
    """相对 log 的 PDMS。轨迹长度应覆盖未来窗口（约 80 帧）。"""
    m = step_metrics(scene, traj)
    hm = scene.human
    T = len(m['coll'])
    if T < 3:
        return {'driving_score': 0.0, 'valid': False}

    nc = 0.0 if np.any(m['coll'] & ~hm['coll']) else 1.0
    dac = 0.0 if np.any(m['off'] & ~hm['off']) else 1.0
    ddc = 0.0 if np.any(m['ddc'] & ~hm['ddc']) else 1.0

    def ratio(bad, good):
        out = np.ones(T)
        valid = np.isfinite(good) & (good > EPS)
        out[valid] = np.clip(good[valid] / np.maximum(bad[valid], EPS), 0.0, 1.0)
        return out

    ttc_s = ratio(m['ttc'], hm['ttc'])
    comfort_s = ratio(m['jerk'], hm['jerk'])
    lk_s = ratio(m['lat'], hm['lat'])
    comfort_s = np.where((hm['jerk'] <= EPS) & (m['jerk'] <= EPS), 1.0, comfort_s)
    comfort_s = np.where((hm['jerk'] <= EPS) & (m['jerk'] > EPS), 0.0, comfort_s)
    lk_s = np.where((hm['lat'] <= EPS) & (m['lat'] <= EPS), 1.0, lk_s)
    lk_s = np.where((hm['lat'] <= EPS) & (m['lat'] > EPS), 0.0, lk_s)

    ep = min(1.0, scene.progress(m['traj'][-1]))
    ttc_avg = float(np.mean(ttc_s))
    comfort_avg = float(np.mean(comfort_s))
    lk_avg = float(np.mean(lk_s))
    weighted = (5 * ep + 5 * ttc_avg + 2 * comfort_avg + 2 * lk_avg) / 14.0
    return {'driving_score': nc * dac * ddc * weighted * 100.0,
            'nc': nc, 'dac': dac, 'ddc': ddc, 'ep': ep,
            'ttc': ttc_avg, 'comfort': comfort_avg, 'lk': lk_avg,
            'collision_steps': int(np.sum(m['coll'] & ~hm['coll'])),
            'offroad_steps': int(np.sum(m['off'] & ~hm['off'])),
            'valid': True}


class SceneCache:
    def __init__(self, data_dir, n_scenes=100, seed=0):
        files = sorted(glob.glob(os.path.join(data_dir, 'tfrecord-*.json')))
        rng = np.random.default_rng(seed)
        if len(files) > n_scenes:
            files = [files[i] for i in rng.choice(len(files), n_scenes, replace=False)]
        self.scenes = []
        for fp in files:
            try:
                with open(fp) as f:
                    self.scenes.append(Scene(json.load(f)))
            except Exception:
                continue
