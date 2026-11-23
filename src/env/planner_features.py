"""规划器特征：离线（JSON）与在线（runtime）共用的唯一实现。

与 build_head_dataset.build_one 的公式逐项一致：
  ego   : v_now(t vs t-2), v_prev(t-5 vs t-7), yaw_rate(t vs t-5), 10 帧历史(自车系)
  agent : 最近 32 个（按 t 时刻距离）：rel(2) + cos/sin(dh) + v + 长宽 + valid + 10 帧历史
  map   : 30 m 内最近的 128 条线段：mid(2) + len + cos/sin + is_lane + valid
"""
import json

import numpy as np

HIST = 10
N_MAX = 32
M_MAX = 128
T_FUT = 8
WP_STEP = 5
T_FUT_EXT = 40
T_NOW = 10


def to_ego(P, p0, h0):
    d = np.asarray(P, np.float64) - np.asarray(p0, np.float64)
    c, s = np.cos(-h0), np.sin(-h0)
    return np.stack([d[..., 0] * c - d[..., 1] * s,
                     d[..., 0] * s + d[..., 1] * c], -1)


class ScenePack:
    """一个场景的全部"日志侧"数据（世界坐标）。"""

    def __init__(self, d):
        objs = d.get('objects') or []
        idx = (d.get('metadata') or {}).get('sdc_track_index')
        sdc = objs[int(idx)]
        self.ego_pos = np.asarray([[q['x'], q['y']] for q in sdc['position']],
                                  np.float64)
        self.ego_head = np.asarray(sdc.get('heading') or [0.0] * len(self.ego_pos),
                                   np.float64)
        self.agents = []
        for i, o in enumerate(objs):
            if i == int(idx) or not o.get('position'):
                continue
            q = o['position']
            if len(q) < HIST + T_FUT + 1:
                continue
            _pos = np.asarray([[p['x'], p['y']] for p in q], np.float64)
            _val = np.asarray(o.get('valid') or [1] * len(q), np.float64)
            # Waymo 用 -10000/无效帧占位：按 valid 掩码把无效帧替换为最近有效位置
            _bad = (_val <= 0.5) | (np.abs(_pos) > 1e4).any(axis=1)
            if _bad.any():
                _ok = np.where(~_bad)[0]
                if len(_ok):
                    _pos[_bad] = _pos[_ok[0]]
                else:
                    _val[:] = 0.0
            self.agents.append(dict(
                pos=_pos,
                head=np.asarray(o.get('heading') or [0.0] * len(q), np.float64),
                valid=_val,
                length=float(o.get('length') or 4.5),
                width=float(o.get('width') or 1.9)))
        A, B, lane = [], [], []
        for r in (d.get('roads') or []):
            typ = r.get('type')
            if typ not in ('lane', 'road_line', 'road_edge') \
                    or len(r.get('geometry') or []) < 2:
                continue
            L = np.asarray([[q['x'], q['y']] for q in r['geometry']], np.float64)
            for a, b in zip(L[:-1], L[1:]):
                A.append(a)
                B.append(b)
                lane.append(1.0 if typ != 'road_edge' else 0.0)
        self.seg_a = np.asarray(A, np.float64) if A else np.zeros((0, 2))
        self.seg_b = np.asarray(B, np.float64) if B else np.zeros((0, 2))
        self.seg_lane = np.asarray(lane, np.float64) if lane else np.zeros((0,))

    def shift(self, dx, dy):
        """整体平移（世界坐标 → GPUDrive sim 坐标，差一个逐场景均值）。"""
        d = np.array([dx, dy], np.float64)
        if len(self.seg_a):
            self.seg_a = self.seg_a + d
            self.seg_b = self.seg_b + d
        for a in self.agents:
            a['pos'] = a['pos'] + d
        if getattr(self, 'ego_pos', None) is not None and len(self.ego_pos):
            self.ego_pos = self.ego_pos + d

    # ---------- ego ----------
    def ego_pose(self, t):
        return self.ego_pos[t], float(self.ego_head[t])

    def hist_poses(self, t, n=HIST):
        """t 及之前 n 帧的世界位姿（含 t），不足则前面重复第一帧。"""
        out = []
        for k in range(t - (n - 1), t + 1):
            kk = max(0, min(k, len(self.ego_pos) - 1))
            out.append((self.ego_pos[kk][0], self.ego_pos[kk][1],
                        float(self.ego_head[kk])))
        return out

    def ego_feat(self, t, hists=None):
        """hists: 10 个世界位姿（最后一个 = 当前自车），默认取日志。"""
        H = np.asarray(hists if hists is not None else self.hist_poses(t),
                       np.float64)
        x, y, h = H[-1]
        v_now = self._spd(H[-1], H[-3] if len(H) >= 3 else H[0], 2)
        v_prev = self._spd(H[4], H[2], 2) if len(H) >= 5 else v_now
        yaw = float(np.arctan2(np.sin(h - H[4][2]),
                               np.cos(h - H[4][2]))) / max(0.1 * 5, 1e-3)
        hist = to_ego(H[:, :2], (x, y), h)
        return np.concatenate([[v_now, v_prev, yaw],
                               hist.reshape(-1)]).astype(np.float32)

    @staticmethod
    def _spd(p1, p0, k):
        return float(np.linalg.norm(np.asarray(p1[:2]) - np.asarray(p0[:2]))
                     / (k * 0.1))

    # ---------- agent ----------
    def agent_feat(self, t, pose):
        x, y, h = pose
        rows = []
        for a in self.agents:
            p = a['pos'][t]
            rows.append((float(np.linalg.norm(p - np.array([x, y]))), a, p))
        rows.sort(key=lambda r: r[0])
        ag = np.zeros((N_MAX, 28), np.float32)
        am = np.zeros(N_MAX, np.float32)
        for n, (_, a, p) in enumerate(rows[:N_MAX]):
            hh = float(a['head'][t])
            dh = hh - h
            v = self._spd(a['pos'][t], a['pos'][max(t - 2, 0)], 2)
            rel = to_ego(p[None], (x, y), h)[0]
            j0 = t - (HIST - 1)
            pts = a['pos'][max(j0, 0):t + 1]
            pe = to_ego(pts, (x, y), h) - rel[None]
            hist = np.zeros((HIST, 2), np.float32)
            hist[-len(pe):] = pe
            ag[n] = np.r_[[rel[0], rel[1], np.cos(dh), np.sin(dh), v,
                           a['length'], a['width'], 1.0], hist.reshape(-1)]
            am[n] = 1.0
        return ag, am

    # ---------- map ----------
    def map_feat(self, pose):
        x, y, h = pose
        if len(self.seg_a) == 0:
            return np.zeros((M_MAX, 7), np.float32), np.zeros(M_MAX, np.float32)
        mid = 0.5 * (self.seg_a + self.seg_b)
        dist = np.linalg.norm(mid - np.array([x, y])[None], axis=1)
        order = np.argsort(dist)[:M_MAX]
        mp = np.zeros((M_MAX, 7), np.float32)
        mm = np.zeros(M_MAX, np.float32)
        for n, i in enumerate(order):
            if dist[i] > 30.0:
                break
            a, b = self.seg_a[i], self.seg_b[i]
            m = to_ego(mid[i][None], (x, y), h)[0]
            ln = float(np.linalg.norm(b - a))
            hh = float(np.arctan2(b[1] - a[1], b[0] - a[0])) - h
            mp[n] = [m[0], m[1], ln, np.cos(hh), np.sin(hh),
                     self.seg_lane[i], 1.0]
            mm[n] = 1.0
        return mp, mm


def load_pack(path):
    with open(path) as fh:
        return ScenePack(json.load(fh))
