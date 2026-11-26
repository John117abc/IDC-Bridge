"""把 84 维 IDC 接进 CARLA agent（plant_agent_vec.py 用）

职责：
  · CARLA 位姿/周车/参考 → canonical 坐标（y 取反、yaw 取反）
  · 参考路径：规划器 8 点（ego 系）→ 世界坐标 + heading/speed/road_dist
  · 调 IDCCarlaPolicy 出 (delta, acc) → CARLA (steer, throttle, brake)
  · 10 Hz 推理（CARLA 20 Hz，中间保持动作）
"""
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import frames as F              # noqa: E402
import idc_carla as IC          # noqa: E402
import idc_state_carla as S     # noqa: E402
import carla_pid_exec as PE     # noqa: E402
import pure_pursuit as PP       # noqa: E402

# CARLA 的 VehicleControl.steer 与我们 canonical(左正) 方向相反，实测需要取负
STEER_SIGN = float(os.environ.get('VEC_IDC_STEER_SIGN', '-1'))


def _load_segments(npz_path):
    """从 map npz 读车道段端点（与 CARLA agent 用同一份地图）。"""
    try:
        z = np.load(npz_path)
        L, NP = z['lanes'], z['lane_pts']
        A, B, o = [], [], 0
        for cnt in NP:
            pl = L[o:o + int(cnt), :2]
            o += int(cnt)
            if len(pl) >= 2:
                A.append(pl[:-1])
                B.append(pl[1:])
        if not A:
            return None, None
        return (np.concatenate(A).astype(np.float64),
                np.concatenate(B).astype(np.float64))
    except Exception as e:                       # noqa: BLE001
        print(f'[IDC] 地图段加载失败: {e}', flush=True)
        return None, None


class IDCController:
    def __init__(self, ckpt, world, vehicle, map_npz, tree=None, infer_every=2,
                 with_policy=True, record_path=''):
        self.pol = IC.IDCCarlaPolicy(ckpt) if with_policy else None
        self.world = world
        self.veh = vehicle
        self.seg_a, self.seg_b = _load_segments(map_npz)
        self.tree = tree
        self.record_path = record_path
        self.rec = []
        self.last_state = None
        self._sv3 = (os.environ.get('VEC_STATE_V3', '0') == '1'
                     or bool(getattr(self.pol, 'v3', False)))
        self.every = max(1, int(infer_every))
        self.cnt = 0
        self.ref_idx = 0
        self.last = (0.0, 0.0, 0.0)
        self.prev_action = (0.0, 0.0)      # (steer, acc) 上一拍，用于状态里的动作历史
        self.prev_v = None
        self.a_lat = 0.0
        self._n_infer = 0
        self._lat = None
        self._lon = None
        self._t_cost = 0.0
        try:
            self.max_steer = float(
                vehicle.get_physics_control().max_steer_angle)
        except Exception:
            self.max_steer = 70.0
        print(f'[IDC] 控制器就绪 ckpt={os.path.basename(ckpt)} '
              f'max_steer={self.max_steer:.0f}° 推理每 {self.every} 帧', flush=True)

    def reset(self):
        self.pol.reset()
        self.cnt = 0
        self.ref_idx = 0
        self.prev_action = (0.0, 0.0)
        self.prev_v = None
        self.last = (0.0, 0.0, 0.0)

    # ---------- 世界 → canonical ----------
    @staticmethod
    def _pose_canon(t):
        if os.environ.get('VEC_IDC_NOMIRROR', '0') == '1':
            # 调试用：不做 y 取反（验证横向符号是否反了）
            return (float(t.location.x), float(t.location.y),
                    math.radians(float(t.rotation.yaw)))
        return F.carla_to_canonical(t.location.x, t.location.y, t.rotation.yaw)

    def _neighbors_canon(self, ego_loc, radius=40.0):
        out = []
        for a in self.world.get_actors().filter('vehicle.*'):
            if a.id == self.veh.id:
                continue
            t = a.get_transform()
            d = math.hypot(t.location.x - ego_loc.x, t.location.y - ego_loc.y)
            if d > radius:
                continue
            v = a.get_velocity()
            sp = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
            cx, cy, ch = (self._pose_canon(t))
            out.append(dict(x=cx, y=cy, h=ch, v=sp, _d=d))
        out.sort(key=lambda q: q['_d'])
        return [{k: v for k, v in q.items() if k != '_d'} for q in out]

    def _route_world(self, route_ego, pose_carla):
        """route（ego 系 (24,4)）→ CARLA 世界坐标 (N,2)。"""
        x, y, h = float(pose_carla[0]), float(pose_carla[1]), float(pose_carla[2])
        c, s = math.cos(h), math.sin(h)
        rp = np.asarray(route_ego, np.float64)
        keep = np.abs(rp[:, 0]) + np.abs(rp[:, 1]) > 0
        rp = rp[keep][:, :2]
        if len(rp) < 2:
            return None
        return np.stack([x + rp[:, 0] * c - rp[:, 1] * s,
                         y + rp[:, 0] * s + rp[:, 1] * c], 1)

    def _clamp_to_corridor(self, pos, route_w, half_w=2.0):
        """把路径点横向偏移限制在 route 走廊内（保证参考可行）。"""
        if route_w is None or len(route_w) < 2:
            return pos
        out = pos.copy()
        for i in range(len(pos)):
            d = np.linalg.norm(route_w - pos[i][None], axis=1)
            j = int(np.argmin(d))
            j0 = max(0, j - 1); j1 = min(len(route_w) - 1, j + 1)
            t = route_w[j1] - route_w[j0]
            nrm = np.linalg.norm(t)
            if nrm < 1e-6:
                continue
            t = t / nrm
            n = np.array([-t[1], t[0]])
            off = float(np.dot(pos[i] - route_w[j], n))
            off_c = float(np.clip(off, -half_w, half_w))
            out[i] = route_w[j] + n * off_c
        return out

    def _path_canon(self, plan_ego, pose_carla, route_ego=None):
        """规划器 8 点（ego 系）→ canonical 世界路径（可选走廊约束）。

        VEC_REF_FROM=route 时：路径形状取自 route（长且平缓），
        规划器只贡献**横向偏移**——避免"短规划 → 曲率虚高 → 限速过低"。
        """
        # ---- plan2：外推 + 样条（开关，默认 legacy）----
        if os.environ.get('VEC_REF_MODE', 'legacy') == 'plan2':
            try:
                pp = S.build_path_ext(
                    np.asarray(plan_ego, np.float64), pose_carla,
                    target_len=float(os.environ.get('VEC_PLAN2_LEN', '28')),
                    v_floor=float(os.environ.get('VEC_PLAN2_VFLOOR', '3.0')),
                    v_max=float(os.environ.get('VEC_IDC_VMAX', '14')),
                    a_lat=float(os.environ.get('VEC_IDC_ALAT', '2.0')))
                pos = np.asarray(pp['pos'], np.float64)
                hd = np.unwrap(np.asarray(pp['heading'], np.float64))
                spd = np.asarray(pp['speed'], np.float32)
                road = np.asarray(pp['road_dist'], np.float32)
                pos_c = np.stack([pos[:, 0], -pos[:, 1]], 1)
                hd_c = (-hd).astype(np.float32)
                if self.tree is not None:
                    try:
                        _d, _ = self.tree.query(pos, k=1)
                        road = np.minimum(np.asarray(_d, np.float32), 200.0)
                    except Exception:
                        pass
                if getattr(self, '_plan2_n', 0) % 20 == 1:
                    print('[PLAN2] len=%.1f kmax=%.3f v=[%.1f,%.1f] plan_v=%.2f'
                          % (pp['path_len'], float(np.max(pp['curv_ahead'])),
                             float(spd.min()), float(spd.max()),
                             pp['plan_speed']), flush=True)
                self._plan2_n = getattr(self, '_plan2_n', 0) + 1
                return dict(pos=pos_c.astype(np.float32), heading=hd_c,
                            speed=spd, road_dist=road,
                            curv_ahead=pp['curv_ahead'],
                            spd_ahead=pp['spd_ahead'],
                            road_ahead=pp['road_ahead'],
                            last_valid_idx=len(pos_c) - 1)
            except Exception as _e:              # noqa: BLE001
                print('[PLAN2] 失败 %s → 回退 legacy' % _e, flush=True)

        if (os.environ.get('VEC_REF_FROM', 'plan') == 'route'
                and route_ego is not None):
            rw = self._route_world(route_ego, pose_carla)
            if rw is not None and len(rw) >= 4:
                # 规划器的横向偏移：取 8 点相对 route 的中位偏移，截断 ±2 m
                try:
                    P = np.asarray(plan_ego, np.float64).reshape(-1, 2)
                    x, y, h = pose_carla[:3]
                    c, s = math.cos(h), math.sin(h)
                    Pw = np.stack([x + P[:, 0] * c - P[:, 1] * s,
                                   y + P[:, 0] * s + P[:, 1] * c], 1)
                    offs = []
                    for q in Pw:
                        d = np.linalg.norm(rw - q[None], axis=1)
                        j = int(np.argmin(d))
                        j0, j1 = max(0, j - 1), min(len(rw) - 1, j + 1)
                        t = rw[j1] - rw[j0]
                        n = np.array([-t[1], t[0]])
                        nn = np.linalg.norm(n)
                        if nn < 1e-6:
                            continue
                        offs.append(float(np.dot(q - rw[j], n / nn)))
                    off = float(np.clip(np.median(offs), -2.0, 2.0)) if offs else 0.0
                except Exception:
                    off = 0.0
                pos = PE.shift_path(rw, off)
                # 1 m 重采样
                d = np.linalg.norm(np.diff(pos, axis=0), axis=1)
                s_arc = np.r_[0.0, np.cumsum(d)]
                if s_arc[-1] > 3.0:
                    ss = np.arange(0.0, s_arc[-1], 1.0)
                    pos = np.stack([np.interp(ss, s_arc, pos[:, 0]),
                                    np.interp(ss, s_arc, pos[:, 1])], 1)
                dd = np.diff(pos, axis=0)
                hd = np.r_[np.unwrap(np.arctan2(dd[:, 1], dd[:, 0])), 0.0]
                hd[-1] = hd[-2] if len(hd) > 1 else 0.0
                hd_c = (-hd).astype(np.float32)
                pos_c = np.stack([pos[:, 0], -pos[:, 1]], 1)
                road = np.asarray([200.0] * len(pos), np.float32)
                if self.tree is not None:
                    try:
                        _d, _ = self.tree.query(pos, k=1)
                        road = np.minimum(np.asarray(_d, np.float32), 200.0)
                    except Exception:
                        pass
                # 曲率限速（1 m 间距，2 m 基线）
                kap = np.zeros(len(pos), np.float64)
                for i in range(len(pos)):
                    j = min(i + 2, len(pos) - 1)
                    ds = max(np.linalg.norm(pos[j] - pos[i]), 1e-3)
                    dth = abs(math.atan2(math.sin(hd[j] - hd[i]),
                                         math.cos(hd[j] - hd[i])))
                    kap[i] = dth / ds
                a_lat = float(os.environ.get('VEC_IDC_ALAT', '2.0'))
                vmax = float(os.environ.get('VEC_IDC_VMAX', '20.0'))
                vmin = float(os.environ.get('VEC_IDC_VMIN', '0.0'))
                spd = np.clip(np.sqrt(a_lat / np.maximum(kap, 1e-4)),
                              vmin, vmax).astype(np.float32)
                try:
                    if not hasattr(self, '_diag_n'):
                        self._diag_n = 0
                    self._diag_n += 1
                    if self._diag_n % 10 == 1:
                        _in_y = float(np.max(np.abs(np.asarray(plan_ego)[:, 1])))
                        _out_y = float(np.max(np.abs(pos[:, 1])))
                        print(f'[PATHDIAG] n={self._diag_n} plan_y={_in_y:+.2f} '
                              f'path_y={_out_y:+.2f}', flush=True)
                except Exception:
                    pass
                return dict(pos=pos_c.astype(np.float32), heading=hd_c,
                            speed=spd, road_dist=road,
                            curv_ahead=spd, spd_ahead=spd, road_ahead=road,
                            last_valid_idx=len(pos_c) - 1)
        # 传 None 跳过 S.build_path 内部的暴力 road_dist，下面用 KD-tree 算
        p = S.build_path(np.asarray(plan_ego, np.float64), pose_carla, None, None)
        pos = np.asarray(p['pos'], np.float64)
        # 走廊约束（保证参考在 route 附近，避免规划跑偏导致撞墙）
        half_w = float(os.environ.get('VEC_IDC_CORRIDOR', '2.0'))
        if half_w > 0 and route_ego is not None:
            try:
                pw = self._clamp_to_corridor(
                    pos, self._route_world(route_ego, pose_carla), half_w)
                pos = pw
            except Exception as _e:               # noqa: BLE001
                print(f'[IDC] 走廊约束失败: {_e}', flush=True)
        pos_c = np.stack([pos[:, 0], -pos[:, 1]], 1)
        hd_c = (-np.asarray(p['heading'], np.float64)).astype(np.float32)
        if os.environ.get('VEC_IDC_NOMIRROR', '0') == '1':
            pos_c = np.stack([pos[:, 0], pos[:, 1]], 1)
            hd_c = np.asarray(p['heading'], np.float32)
        road = np.asarray(p['road_dist'], np.float32)
        # 参考速度：与训练同口径的**曲率限速**（不用规划器点间距，规划器常给出爬行轨迹）
        #   v = clip(sqrt(a_lat / κ), v_min, v_max)，κ 用固定 2 m 基线的中心差分
        #   （直接用相邻点会因 spline 起点重复点导致 ds≈0、κ 爆炸）
        hd = np.unwrap(np.asarray(p['heading'], np.float64))
        s_arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pos, axis=0), axis=1))]
        n_pt = len(pos)
        base = 2.0
        kap = np.zeros(n_pt, np.float64)
        for i in range(n_pt):
            j = int(np.searchsorted(s_arc, s_arc[i] + base))
            if j >= n_pt:
                j = n_pt - 1
            ds = max(s_arc[j] - s_arc[i], 1e-3)
            dth = abs(math.atan2(math.sin(hd[j] - hd[i]),
                                 math.cos(hd[j] - hd[i])))
            kap[i] = dth / ds
        a_lat = float(os.environ.get('VEC_IDC_ALAT', '2.0'))
        vmax = float(os.environ.get('VEC_IDC_VMAX', '20.0'))
        vmin = float(os.environ.get('VEC_IDC_VMIN', '0.0'))
        spd = np.clip(np.sqrt(a_lat / np.maximum(kap, 1e-4)), vmin, vmax).astype(np.float32)
        if self.tree is not None:
            try:
                _d, _ = self.tree.query(np.asarray(p['pos'], np.float64), k=1)
                road = np.minimum(np.asarray(_d, np.float32), 200.0)
            except Exception as _e:               # noqa: BLE001
                print(f'[IDC] road_dist 查询失败: {_e}', flush=True)
        return dict(pos=pos_c.astype(np.float32), heading=hd_c,
                    speed=spd,
                    road_dist=road,
                    curv_ahead=np.asarray(p['curv_ahead'], np.float32),
                    spd_ahead=np.asarray(p['spd_ahead'], np.float32),
                    road_ahead=np.asarray(p['road_ahead'], np.float32),
                    last_valid_idx=len(pos_c) - 1)

    def _vru_canon(self, t, horizon=40.0):
        """行人/自行车（canonical：y 取反）。"""
        out = []
        x, y = t.location.x, t.location.y
        try:
            for a in self.world.get_actors().filter('walker.pedestrian.*'):
                at = a.get_transform()
                dx, dy = at.location.x - x, at.location.y - y
                if math.hypot(dx, dy) > horizon:
                    continue
                av = a.get_velocity()
                out.append(dict(x=float(at.location.x),
                                y=float(-at.location.y),
                                vx=float(av.x), vy=float(-av.y)))
        except Exception:
            return out
        return out

    def _light_feat(self, t, horizon=50.0):
        """最近的相关交通灯 → (红/黄/绿 one-hot + 距离)。"""
        x, y = t.location.x, t.location.y
        h = math.radians(t.rotation.yaw)
        c, s = math.cos(h), math.sin(h)
        best = None
        try:
            for a in self.world.get_actors().filter('traffic.traffic_light*'):
                at = a.get_transform()
                dx, dy = at.location.x - x, at.location.y - y
                lx = dx * c + dy * s
                ly = -dx * s + dy * c
                if lx <= 0.0 or lx > horizon or abs(ly) > 6.0:
                    continue
                d = math.hypot(dx, dy)
                if best is None or d < best[0]:
                    best = (d, str(a.state))
        except Exception:
            return F.light_features(0.0, None)
        if best is None:
            return F.light_features(0.0, None)
        return F.light_features(best[0], best[1])

    def _safety_level(self, t, speed, half_w=3.0, horizon=25.0):
        """分级安全等级 0..1（0=不干预，1=全力刹）。

        车辆：按 TTC 线性升；
        行人：按距离分级（8 m 开始，3 m 全力）；
        静态障碍（锥桶/围挡）：6 m 开始，2.5 m 全力。
        """
        if os.environ.get('VEC_SAFETY', '1') != '1':
            return 0.0
        x, y = t.location.x, t.location.y
        h = math.radians(t.rotation.yaw)
        c, s = math.cos(h), math.sin(h)
        v_ego = max(float(speed), 0.0)
        level = 0.0

        # --- 车辆：TTC ---
        ttc_thr = float(os.environ.get('VEC_SAFE_TTC', '3.5'))
        for a in self.world.get_actors().filter('vehicle.*'):
            if a.id == self.veh.id:
                continue
            at = a.get_transform()
            dx, dy = at.location.x - x, at.location.y - y
            lx = dx * c + dy * s
            ly = -dx * s + dy * c
            if lx <= 0.5 or lx > horizon or abs(ly) > half_w:
                continue
            av = a.get_velocity()
            v_o = math.sqrt(av.x ** 2 + av.y ** 2 + av.z ** 2)
            closing = v_ego - v_o * 0.5
            ttc = lx / max(closing, 0.3)
            if ttc < ttc_thr:
                level = max(level, float(np.clip((ttc_thr - ttc) / 2.0, 0.0, 1.0)))

        # --- 行人：按距离分级 ---
        ped_far = float(os.environ.get('VEC_SAFE_PED_FAR', '8.0'))
        ped_near = float(os.environ.get('VEC_SAFE_PED_NEAR', '3.0'))
        for a in self.world.get_actors().filter('walker.pedestrian.*'):
            at = a.get_transform()
            dx, dy = at.location.x - x, at.location.y - y
            lx = dx * c + dy * s
            ly = -dx * s + dy * c
            if lx <= 0.0 or lx > ped_far or abs(ly) > half_w:
                continue
            lvl = float(np.clip((ped_far - lx) / max(ped_far - ped_near, 0.5), 0.0, 1.0))
            level = max(level, lvl)

        # --- 静态障碍（施工锥桶/围挡/标志）---
        sp_far = float(os.environ.get('VEC_SAFE_STATIC_FAR', '7.0'))
        sp_near = float(os.environ.get('VEC_SAFE_STATIC_NEAR', '2.5'))
        try:
            for a in self.world.get_actors().filter('static.prop.*'):
                at = a.get_transform()
                dx, dy = at.location.x - x, at.location.y - y
                lx = dx * c + dy * s
                ly = -dx * s + dy * c
                if lx <= 0.0 or lx > sp_far or abs(ly) > 2.5:
                    continue
                lvl = float(np.clip((sp_far - lx) / max(sp_far - sp_near, 0.5), 0.0, 1.0))
                level = max(level, lvl)
        except Exception:
            pass

        return float(np.clip(level, 0.0, 1.0))

    def control(self, pose_carla, plan_ego, speed, route_ego=None):
        """返回 (steer, throttle, brake)。with_policy=False 时只构建状态（录制用）。"""
        self.cnt += 1
        # 卡死即终止（任何执行模式都生效，避免白等整条路线）
        _lim = float(os.environ.get('VEC_STUCK_SEC', '0'))
        if _lim > 0:
            if speed < 0.2:
                self._stuck_s = getattr(self, '_stuck_s', 0.0) + 0.05
            else:
                self._stuck_s = 0.0
            if self._stuck_s > _lim:
                # 卡死诊断：周围有什么 + route 横向偏移
                try:
                    t = self.veh.get_transform()
                    _st, _vh = [], []
                    for a in self.world.get_actors().filter('static.prop.*'):
                        at = a.get_transform()
                        d = float(np.hypot(at.location.x - t.location.x,
                                           at.location.y - t.location.y))
                        if d < 15.0:
                            _st.append((round(d, 2), a.type_id.split('.')[-1]))
                    for a in self.world.get_actors().filter('vehicle.*'):
                        if a.id == self.veh.id:
                            continue
                        at = a.get_transform()
                        d = float(np.hypot(at.location.x - t.location.x,
                                           at.location.y - t.location.y))
                        if d < 15.0:
                            _vh.append((round(d, 2), a.type_id.split('.')[-1]))
                    _st.sort(); _vh.sort()
                    print(f'[STUCK-DIAG] 静态物={_st[:5]}', flush=True)
                    print(f'[STUCK-DIAG] 车辆={_vh[:5]}', flush=True)
                    print(f'[STUCK-DIAG] route 首点横向={float(route_ego[0, 1]):+.2f} m'
                          if route_ego is not None else '[STUCK-DIAG] 无 route', flush=True)
                except Exception as _e:               # noqa: BLE001
                    print(f'[STUCK-DIAG] 诊断失败: {_e}', flush=True)
                print(f'[IDC] STUCK 车速<0.2 已 {self._stuck_s:.1f}s → 立即终止',
                      flush=True)
                raise RuntimeError('STUCK_EARLY_ABORT')
        if self.cnt % self.every == 0 or self.cnt == 1:
            t = self.veh.get_transform()
            ego_c = self._pose_canon(t)
            nbrs = self._neighbors_canon(t.location)
            path = self._path_canon(plan_ego, pose_carla, route_ego)
            _vru = (F.vru_summary(ego_c, self._vru_canon(t), v_ego=float(speed))
                    if self._sv3 else None)
            _light = self._light_feat(t) if self._sv3 else None
            _edyn = None
            _cmd = None
            if self._sv3:
                try:
                    _wz = float(self.veh.get_angular_velocity().z)
                except Exception:
                    _wz = 0.0
                _sp = float(self.prev_action[0]) if self.prev_action else 0.0
                _edyn = F.ego_dyn_features(_sp, getattr(self, 'a_lat', 0.0), -_wz)
                _i15 = int(min(self.ref_idx + 15, len(path['pos']) - 1))
                _cmd = F.cmd_features(float(ego_c[2]),
                                      float(path['heading'][_i15]))
            st, self.ref_idx = IC.make_state84(
                ego_c, float(speed), nbrs, path, ref_idx=self.ref_idx,
                veh_len=self.veh.bounding_box.extent.x * 2.0,
                # 训练时这两项恒为 0，部署必须一致，否则输入 OOD
                a_lat=0.0, steer_prev=0.0,
                v3=self._sv3, vru=_vru, light=_light,
                ego_dyn=_edyn, cmd=_cmd, remain=0.0)
            self.last_state = st.copy()
            if self.pol is None:            # 录制模式：只出状态，动作由外部给
                return self.last
            delta, acc = self.pol.act(st)
            self._n_infer += 1
            # ---- 执行层：PID（PlanT 口径）----
            #   lateral: 用 IDC 的曲率意图把 route 横移，再用 PD+前视跟踪
            #   longitudinal: v_ref + acc 修正 作为目标速度，PID 出油门/刹车
            if os.environ.get('VEC_EXEC', 'pid') == 'pid':
                if self._lat is None:
                    self._lat, self._lon = PE.LateralPID(), PE.LongitudinalPID()
                if getattr(self, '_pp', None) is None:
                    self._pp = PP.PurePursuit()
                t = self.veh.get_transform()
                wpos = (t.location.x, t.location.y)
                wh = math.radians(t.rotation.yaw)
                route_w = self._route_world(route_ego, pose_carla) if route_ego is not None else None
                d_prev = float(os.environ.get('VEC_IDC_PREVIEW', '5.0'))
                off_gain = float(os.environ.get('VEC_IDC_OFF_GAIN', '0.0'))
                if getattr(self.pol, 'action_space', 'delta_acc') == 'off_acc':
                    # 动作空间 2D：网络直接输出 (横向偏移 m, 加速度 m/s²)
                    _omax = float(os.environ.get('VEC_IDC_OFF_MAX',
                                                 self.pol.off_max))
                    off = float(np.clip(delta, -_omax, _omax))
                    acc = float(np.clip(acc, -3.0, 1.5))
                else:
                    off = -off_gain * d_prev * math.tan(float(delta))
                if route_w is not None:
                    _path = PE.shift_path(route_w, off)
                    # ---- TK-OFF2：直接改将被纯追踪消费的 _path ----
                    try:
                        _th = int(getattr(self, 'takeover_hold', 0) or 0)
                        if _th > 0 and os.environ.get('VEC_TAKEOVER', '0') == '1':
                            _off = float(np.clip(getattr(self, 'takeover_off', 0.0),
                                                 -4.0, 4.0))
                            # ---- v0.7 三段式：借出(2~14m) → 保持(14~26m) → 并回(26~38m) ----
                            _P = np.asarray(route_w, np.float64)
                            _dd = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(_P, axis=0), axis=1))]
                            _j0 = int(np.argmin(np.linalg.norm(_P - np.array(wpos), axis=1)))
                            _ds = _dd - _dd[_j0]
                            _bar = getattr(self, 'takeover_bar', None)
                            _b0, _b1 = (float(_bar[0]), float(_bar[1])) if _bar else (8.0, 16.0)
                            _r0s = max(0.0, _b0 - 8.0)     # 借出起点
                            _r1s = max(_r0s + 1.0, _b0)    # 到达满偏移
                            _hs = _b1 + 2.0                # 保持结束
                            _he = _hs + 10.0               # 并回结束
                            _o = np.zeros(len(_P))
                            for _i, _s in enumerate(_ds):
                                if _r0s <= _s < _r1s:
                                    _u = (_s - _r0s) / (_r1s - _r0s)
                                    _o[_i] = _off * (3 * _u ** 2 - 2 * _u ** 3)
                                elif _r1s <= _s < _hs:
                                    _o[_i] = _off
                                elif _hs <= _s < _he:
                                    _u = (_s - _hs) / (_he - _hs)
                                    _o[_i] = _off * (1 - (3 * _u ** 2 - 2 * _u ** 3))
                            _path = PE.shift_path(_P, _o)
                            self.takeover_hold = _th - 1
                            if _th % 10 == 0:
                                print('[TK-PROF] off=%+.2f 剩余=%d 剖面max=%.2f' % (
                                    _off, _th, float(np.max(np.abs(_o)))), flush=True)
                    except Exception as _e:              # noqa: BLE001
                        print('[TK-OFF2] 失败 %s' % _e, flush=True)
                    # ---- TK-PROFILE：借出→保持→并回 ----
                    try:
                        if (int(getattr(self, 'takeover_hold', 0) or 0) > 0
                                and int(getattr(self, 'takeover_profile', 0) or 0) == 1
                                and os.environ.get('VEC_TAKEOVER', '0') == '1'):
                            _dy = float(np.clip(getattr(self, 'takeover_off', 0.0), -4.0, 4.0))
                            _P = np.asarray(route_w, np.float64)
                            _d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(_P, axis=0), axis=1))]
                            _j0 = int(np.argmin(np.linalg.norm(_P - np.array(wpos), axis=1)))
                            _ds = _d - _d[_j0]
                            _o = np.zeros(len(_P))
                            for _i, _s in enumerate(_ds):
                                if 2.0 <= _s < 14.0:
                                    _u = (_s - 2.0) / 12.0
                                    _o[_i] = _dy * (3 * _u ** 2 - 2 * _u ** 3)
                                elif 14.0 <= _s < 26.0:
                                    _o[_i] = _dy
                                elif 26.0 <= _s < 38.0:
                                    _u = (_s - 26.0) / 12.0
                                    _o[_i] = _dy * (1 - (3 * _u ** 2 - 2 * _u ** 3))
                            _path = PE.shift_path(_P, _o)
                            self.takeover_hold = int(getattr(self, 'takeover_hold', 0)) - 1
                            if self.takeover_hold % 10 == 0:
                                print('[TK-PROF] max_off=%.2f 剩余=%d' % (
                                    float(np.max(np.abs(_o))), self.takeover_hold), flush=True)
                    except Exception as _e:              # noqa: BLE001
                        print('[TK-PROF] 失败 %s' % _e, flush=True)
                    if os.environ.get('VEC_LAT_MODE', 'pp') == 'pp':
                        steer = self._pp.step(_path, float(speed), wpos, wh,
                                              dt=0.05 * self.every)
                        try:
                            if not hasattr(self, '_tkdbg_n'):
                                self._tkdbg_n = 0
                            self._tkdbg_n += 1
                            if self._tkdbg_n % 4 == 1:
                                _dp = np.asarray(_path, np.float64)
                                _dr = np.asarray(route_w, np.float64)
                                _j = int(np.argmin(
                                    np.linalg.norm(_dr - np.array(wpos), axis=1)))
                                _k = min(_j + 6, len(_dp) - 1, len(_dr) - 1)
                                _toff = float(np.linalg.norm(_dp[_k] - _dr[_k]))
                                print('[TK-DBG] n=%d off=%+.3f steer=%+.3f '
                                      'tgt_dy=%.2f' % (self._tkdbg_n, off, steer,
                                                       _toff), flush=True)
                        except Exception as _e:              # noqa: BLE001
                            print('[TK-DBG] 失败 %s' % _e, flush=True)
                    else:
                        steer = self._lat.step(_path, float(speed), wpos, wh)
                else:
                    steer = 0.0
                _i5 = int(min(self.ref_idx + 5, len(path['speed']) - 1))
                _vg = float(os.environ.get('VEC_IDC_ACC_GAIN', '0.15'))
                # 前方窗口取最小值（为最紧的弯减速）+ 帧间低通，避免 v_tgt 抖动
                _j1 = int(min(_i5 + 8, len(path['speed']) - 1))
                _v_raw = float(np.min(path['speed'][_i5:_j1 + 1]))
                if getattr(self.pol, 'action_space', 'delta_acc') == 'off_acc':
                    # 2D 动作空间
                    _vmin = float(os.environ.get('VEC_IDC_VMIN', '0.0'))
                    if os.environ.get('VEC_IDC_LONG_REF', '1') == '1':
                        # 参考速度 + acc 小幅修正（不积分，避免 acc≈0 时永远起步不了）
                        _v_ref0 = float(_v_raw)
                        _v_raw = float(np.clip(_v_ref0 + _vg * float(acc),
                                               _vmin, 20.0))
                    else:
                        _dt = 0.05 * self.every
                        _vc = float(getattr(self, '_v_cmd', None) or _vmin)
                        _vc = float(np.clip(_vc + acc * _dt, _vmin, 20.0))
                        self._v_cmd = _vc
                        _v_raw = _vc
                else:
                    _v_raw = float(np.clip(_v_raw + _vg * float(acc), 0.0, 20.0))
                _prev = getattr(self, '_v_tgt_prev', None)
                _a = float(os.environ.get('VEC_VTGT_A', '0.9'))
                v_tgt = _v_raw if _prev is None else _a * _prev + (1 - _a) * _v_raw
                if _prev is not None:
                    _dmax = float(os.environ.get('VEC_VTGT_DMAX', '0.8'))
                    v_tgt = float(np.clip(v_tgt, _prev - _dmax, _prev + _dmax))
                _cap = float(os.environ.get('VEC_V_CAP', '0') or 0)
                if _cap > 0:
                    v_tgt = min(v_tgt, _cap)
                self._v_tgt_prev = v_tgt
                thr, brk = self._lon.step(v_tgt, float(speed),
                                          dt=0.05 * self.every)
                # 分级安全层：按等级压油门 + 加刹车
                _lvl = self._safety_level(t, float(speed))
                if _lvl > 0.02:
                    thr = float(thr) * (1.0 - _lvl)
                    brk = max(float(brk), _lvl)
                self.prev_v = float(speed)
                self.last = (steer, thr, 1.0 if brk else 0.0)
                if self.cnt % 20 == 0:
                    print(f'[PID] v={speed:.2f} v_tgt={v_tgt:.2f} '
                          f'steer={steer:+.2f} thr={thr:.2f} brk={brk:.2f} '
                          f'off={off:+.2f} d={delta:+.3f}', flush=True)
                return self.last
            # 纵向可由 PI 接管（VEC_IDC_LONG=pi）：BC 主要学到转向，
            # 纵向用 PI 跟踪 v_ref 更稳，避免“刹死不动”
            if os.environ.get('VEC_IDC_LONG', 'idc') == 'pi':
                _i5 = int(min(self.ref_idx + 5, len(path['speed']) - 1))
                v_ref = float(path['speed'][_i5])
                err = v_ref - float(speed)
                self._spd_i = float(np.clip(
                    getattr(self, '_spd_i', 0.0) + err * 0.05, -3.0, 8.0))
                acc = float(np.clip(0.55 * err + 0.30 * self._spd_i, -3.0, 1.5))
            steer, thr, brk = IC.to_carla_control(
                delta, acc, max_steer_deg=self.max_steer,
                steer_gain=float(os.environ.get('VEC_IDC_STEER_GAIN', '1.0')))
            steer *= STEER_SIGN
            if self.prev_v is not None:
                self.a_lat = (float(speed) - self.prev_v) / 0.1
            self.prev_v = float(speed)
            self.prev_action = (steer, acc)
            self.last = (steer, thr, brk)
            if os.environ.get('VEC_IDC_DEBUG', '0') == '1' and self._n_infer % 20 == 0:
                dp, dphi, dv = float(st[46]), float(st[47]), float(st[48])
                vref = float(path['speed'][min(self.ref_idx + 5,
                                               len(path['speed']) - 1)])
                print(f'[IDCdbg] n={self._n_infer} v={speed:.2f} vref={vref:.2f} '
                      f'dp={dp:+.2f} dphi={dphi:+.3f} dv={dv:+.2f} '
                      f'-> delta={delta:+.3f} acc={acc:+.2f} '
                      f'-> steer={steer:+.3f} thr={thr:.2f} brk={brk:.2f}',
                      flush=True)
        return self.last

    # ---------- 录制（纯追踪当老师，供 BC 微调）----------
    def record(self, steer, thr, brk):
        if self.record_path and self.last_state is not None:
            self.rec.append((self.last_state.copy(),
                             np.array([steer, thr, brk], np.float32)))
            if len(self.rec) >= 800:          # 增量落盘，避免被杀进程丢数据
                self.flush()

    def flush(self):
        if self.record_path and self.rec:
            os.makedirs(os.path.dirname(self.record_path), exist_ok=True)
            S = np.stack([r[0] for r in self.rec])
            A = np.stack([r[1] for r in self.rec])
            _i = getattr(self, '_flush_id', 0)
            self._flush_id = _i + 1
            _base = getattr(self, '_rec_base', self.record_path)
            self._rec_base = _base
            _p = _base.replace('.npz', f'_{_i}.npz')
            np.savez_compressed(_p, state=S, action=A)
            print(f'[BC] 保存 {S.shape} -> {self.record_path}', flush=True)
            self.rec = []
