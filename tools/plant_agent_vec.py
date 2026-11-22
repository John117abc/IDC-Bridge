"""M3-2：向量规划器接入 CARLA 闭环（自包含实现，不依赖 PlanT2 的 BEV 栈）。

输入（训练口径）：
  ego 10 帧历史 / 周车(位置,朝向,速度,长宽) / 地图(我们导出的 npz) / route(leaderboard global plan)
输出：8 点轨迹（ego 系，0.5 s）→ 置信度 argmax 选模态 → 三次样条密化 41 点 → 纯追踪 + 速度 P 控制
"""
import os
import sys

import numpy as np
import torch
import carla

from leaderboard.autoagents import autonomous_agent
from scipy.interpolate import CubicSpline

sys.path.insert(0, '/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/src')
sys.path.insert(0, '/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/tools')
from train_planner import Planner      # noqa: E402

MAPS = '/mnt/2T_HDD/plant2_vector/maps'
CKPT = os.environ.get('VEC_CKPT', '/mnt/2T_HDD/plant2_vector/ckpt_conf/best.pt')
ROUTE_ONLY = os.environ.get('VEC_ROUTE_ONLY', '0') == '1'
REPLAN_EVERY = max(1, int(os.environ.get('VEC_REPLAN', '5')))
REC = os.environ.get('VEC_RECORD', '0') == '1'
REC_DIR = '/mnt/2T_HDD/plant2_vector/video'
DEBUG = os.environ.get('VEC_DEBUG', '1') == '1'


def get_entry_point():
    return 'VecAgent'


class VecAgent(autonomous_agent.AutonomousAgent):

    def setup(self, path_to_conf_file):
        self.track = autonomous_agent.Track.SENSORS
        self.step = 0
        self.hist = []
        self.dbg = []
        self.map_cache = {}
        self.xml_routes = []
        self._rp = None
        self._plan_world = None
        self._plan_mode = -1
        self._spd_i = 0.0
        self._stuck_t = 0.0
        self._last_thr = 0.0
        try:
            import xml.etree.ElementTree as _ET
            _fp = os.environ.get(
                'VEC_ROUTES',
                '/home/jiangchengxuan/PycharmProjects/research-safe-drive/plant2/'
                'Bench2Drive/leaderboard/data/two_routes.xml')
            for _r in _ET.parse(_fp).getroot().findall('route'):
                _pts = [(float(p.get('x')), float(p.get('y')))
                        for p in _r.find('waypoints').findall('position')]
                if len(_pts) > 10:
                    self.xml_routes.append(np.asarray(_pts, np.float64))
            print(f'[VecAgent] route xml: {len(self.xml_routes)} 条', flush=True)
        except Exception as _e:
            print('[VecAgent] route xml 解析失败:', _e, flush=True)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        ck = torch.load(CKPT, map_location='cpu')
        self.K = int(ck.get('K', 6))
        self.model = Planner(K=self.K).to(self.device)
        self.model.load_state_dict(ck['model'], strict=False)
        self.model.emit_logits = True
        self.model.eval()
        self._vehicle = None
        if REC:
            os.makedirs(REC_DIR, exist_ok=True)
        print(f'[VecAgent] ckpt={CKPT} K={self.K} rec={REC} route_only={ROUTE_ONLY}', flush=True)

    def sensors(self):
        return [{'type': 'sensor.camera.rgb', 'x': 1.3, 'y': 0.0, 'z': 2.3,
                 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                 'width': 640, 'height': 360, 'fov': 90, 'id': 'rgb'}]

    def _find_hero(self):
        try:
            from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
            v = CarlaDataProvider.get_hero_actor()
            if v is not None:
                return v
            w = CarlaDataProvider.get_world()
            if w is not None:
                for a in w.get_actors().filter('vehicle.*'):
                    if a.attributes.get('role_name') == 'hero':
                        return a
        except Exception as e:
            print('[VecAgent] find_hero 失败:', e, flush=True)
        return None

    # ---------------- 特征 ----------------
    def _pose(self):
        t = self._vehicle.get_transform()
        v = self._vehicle.get_velocity()
        return np.array([t.location.x, t.location.y,
                         float(np.deg2rad(t.rotation.yaw)),
                         float(np.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))], np.float64)

    def _map(self, town):
        """加载一次整城地图 + 对线段中点建 KD-tree（后续每帧只做 query）。"""
        if town in self.map_cache:
            return self.map_cache[town]
        fp = os.path.join(MAPS, f'map_{town}.npz')
        if not os.path.exists(fp):
            self.map_cache[town] = None
            return None
        from scipy.spatial import cKDTree
        z = np.load(fp)
        L, NP = z['lanes'], z['lane_pts']
        A, B, o = [], [], 0
        for n in NP:
            pl = L[o:o + n, :2]
            o += n
            if len(pl) >= 2:
                A.append(pl[:-1])
                B.append(pl[1:])
        A = np.concatenate(A).astype(np.float64)
        B = np.concatenate(B).astype(np.float64)
        mid = 0.5 * (A + B)
        d = B - A
        slen = np.linalg.norm(d, axis=1)
        shd = np.arctan2(d[:, 1], d[:, 0])
        tree = cKDTree(mid)
        self.map_cache[town] = (tree, mid, slen, shd)
        print(f'[VecAgent] 地图索引 {town}: {len(mid)} 段', flush=True)
        return self.map_cache[town]

    def _plan_pts(self):
        """把 leaderboard 的全局路线解析成 (N,2) 世界坐标，尽量兼容各种格式。"""
        for attr in ('_global_plan_world_coord', '_global_plan'):
            plan = getattr(self, attr, None)
            if not plan:
                continue
            pts = []
            for item in plan:
                obj = item[0] if isinstance(item, (tuple, list)) and len(item) >= 1 else item
                got = None
                if hasattr(obj, 'location'):
                    try:
                        got = [float(obj.location.x), float(obj.location.y)]
                    except Exception:
                        got = None
                if got is None and isinstance(obj, dict):
                    for kx, ky in (('x', 'y'), ('loc_x', 'loc_y')):
                        if kx in obj and ky in obj:
                            got = [float(obj[kx]), float(obj[ky])]
                            break
                if got is None and isinstance(obj, (tuple, list)) and len(obj) >= 2:
                    try:
                        got = [float(obj[0]), float(obj[1])]
                    except Exception:
                        got = None
                if got is None and isinstance(obj, dict) and 'transform' in obj:
                    try:
                        t = obj['transform']
                        got = [float(t['location']['x']), float(t['location']['y'])]
                    except Exception:
                        got = None
                if got is not None:
                    pts.append(got)
            if len(pts) >= 3:
                a = np.asarray(pts, np.float64)
                if not getattr(self, '_plan_logged', False):
                    print(f'[VecAgent] plan {attr}: {len(a)} 点, 首点 {a[0]}', flush=True)
                    self._plan_logged = True
                return a
        if not getattr(self, '_plan_logged', False):
            print('[VecAgent] 警告：拿不到全局路线', flush=True)
            self._plan_logged = True
        return None

    def _route_from_lane(self, goal):
        """从自车车道出发沿拓扑前进，每个路口选朝向 goal 的分支。"""
        try:
            wmap = self._vehicle.get_world().get_map()
            loc = self._vehicle.get_transform().location
            wp = wmap.get_waypoint(loc, project_to_road=True,
                                   lane_type=carla.LaneType.Driving)
            if wp is None:
                return None
            gx, gy = float(goal[0]), float(goal[1])
            out = [[wp.transform.location.x, wp.transform.location.y]]
            for _ in range(40):                       # 40 * 2 m = 80 m
                nxts = wp.next(2.0)
                if not nxts:
                    break
                bx, by = float(wp.transform.location.x), float(wp.transform.location.y)
                gb = np.arctan2(gy - by, gx - bx)
                best, bd = None, 1e9
                for nw in nxts:
                    yaw = np.deg2rad(nw.transform.rotation.yaw)
                    d = abs(np.arctan2(np.sin(yaw - gb), np.cos(yaw - gb)))
                    if d < bd:
                        bd, best = d, nw
                if best is None:
                    break
                wp = best
                out.append([wp.transform.location.x, wp.transform.location.y])
                if np.hypot(wp.transform.location.x - gx,
                            wp.transform.location.y - gy) < 5.0:
                    break
            return np.asarray(out, np.float64) if len(out) >= 5 else None
        except Exception as e:
            print('[VecAgent] lane route 失败:', e, flush=True)
            return None

    def _densify(self, pts):
        """优先用车道拓扑生成（保证从自车出发、只向前）。"""
        r = self._route_from_lane(pts[-1])
        if r is not None:
            return r
        return self._densify_trace(pts)

    def _densify_trace(self, pts):
        """（旧）从自车 trace 到终点。"""
        try:
            from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
            from agents.navigation.global_route_planner import GlobalRoutePlanner
            import carla as _carla
            m = CarlaDataProvider.get_map()
            if m is None:
                return pts
            if getattr(self, '_grp', None) is None:
                self._grp = GlobalRoutePlanner(m, 2.0)
            ego_loc = self._vehicle.get_transform().location
            goal = pts[-1]
            seg = self._grp.trace_route(
                _carla.Location(x=float(ego_loc.x), y=float(ego_loc.y), z=float(ego_loc.z)),
                _carla.Location(x=float(goal[0]), y=float(goal[1]), z=0.0))
            out = [[wp.transform.location.x, wp.transform.location.y] for wp, _ in seg]
            if len(out) >= 10:
                return np.asarray(out, np.float64)
        except Exception as e:
            print('[VecAgent] trace_route 失败:', e, flush=True)
        return pts

    def _densify_chain(self, pts):
        """（旧）沿稀疏点逐段补全。"""
        if getattr(self, '_dense', None) is not None and len(self._dense) >= 20:
            return self._dense
        try:
            from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
            from agents.navigation.global_route_planner import GlobalRoutePlanner
            import carla as _carla
            m = CarlaDataProvider.get_map()
            if m is None:
                return pts
            if getattr(self, '_grp', None) is None:
                self._grp = GlobalRoutePlanner(m, 2.0)
            out = []
            for a, b in zip(pts[:-1], pts[1:]):
                try:
                    seg = self._grp.trace_route(
                        _carla.Location(x=float(a[0]), y=float(a[1]), z=0.0),
                        _carla.Location(x=float(b[0]), y=float(b[1]), z=0.0))
                    for wp, _opt in seg:
                        out.append([wp.transform.location.x, wp.transform.location.y])
                except Exception:
                    continue
            if len(out) >= 10:
                self._dense = np.asarray(out, np.float64)
                print(f'[VecAgent] route 稠密化 {len(pts)} -> {len(self._dense)} 点', flush=True)
                return self._dense
        except Exception as e:
            print('[VecAgent] 稠密化失败:', e, flush=True)
        return pts

    def _dense_route(self, plan):
        """优先用 route XML 的稠密路点（按起点匹配，避免图和拓扑推断）。"""
        if self.xml_routes and plan is not None and len(plan) > 0:
            p0 = np.asarray(plan[0], np.float64)
            best, bd = None, 1e9
            for r in self.xml_routes:
                d = float(np.linalg.norm(r[0] - p0))
                if d < bd:
                    bd, best = d, r
            if best is not None and bd < 20.0:
                return best
        return None

    def _route(self, pose):
        """route（优先 XML 稠密点）→ ego 系 24 点。"""
        plan = self._plan_pts()
        pts = self._dense_route(plan)
        if pts is None:
            if plan is None or len(plan) < 2:
                return np.zeros((24, 4), np.float32)
            pts = self._densify(plan)
        x, y, h, _ = pose
        # 用 CARLA 的逆矩阵做世界→ego 系（局部系 x 前、y 右；训练是 y 左 → 取负）
        c_, s_ = np.cos(-h), np.sin(-h)
        d = pts - np.array([x, y])
        e = np.stack([d[:, 0] * c_ - d[:, 1] * s_, d[:, 0] * s_ + d[:, 1] * c_], 1)
        # 从"离自车最近的点"开始取，保证 route 起点在自车附近（与训练同口径）
        j0 = int(np.argmin(np.linalg.norm(e, axis=1)))
        e = e[j0:]
        if len(e) < 3:
            return np.zeros((24, 4), np.float32)
        s_ = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(e, axis=0), axis=1))]
        keep = s_ <= 48.0
        e, s_ = e[keep], s_[keep]
        if len(e) < 3:
            return np.zeros((24, 4), np.float32)
        ss = np.arange(0.0, min(48.0, s_[-1]), 2.0)
        n = min(24, len(ss))
        out = np.zeros((24, 4), np.float32)
        xs = np.interp(ss[:n], s_, e[:, 0])
        ys = np.interp(ss[:n], s_, e[:, 1])
        out[:n, 0], out[:n, 1], out[:n, 2] = xs, ys, ss[:n]
        hd = np.arctan2(np.diff(ys), np.diff(xs))
        out[:max(n - 1, 0), 3] = np.cos(hd)
        out[n - 1, 3] = out[max(n - 2, 0), 3]
        if not getattr(self, '_route_logged', False):
            print('[VecAgent] route 特征: n=%d 最远 %.1f m 横向范围 [%.1f, %.1f]'
                  % (n, float(ss[n - 1]), float(ys.min()), float(ys.max())), flush=True)
            self._route_logged = True
        return out

    def _feat(self, pose):
        x, y, h, spd = pose
        self.hist.append([x, y, h])
        self.hist = self.hist[-10:]
        while len(self.hist) < 10:
            self.hist.insert(0, self.hist[0])
        H = np.asarray(self.hist)
        c, s = np.cos(-h), np.sin(-h)
        d = H[:, :2] - np.array([x, y])
        hist = np.stack([d[:, 0] * c - d[:, 1] * s, d[:, 0] * s + d[:, 1] * c], 1)
        v_prev = float(np.linalg.norm(H[-1, :2] - H[-3, :2]) / 0.2)
        dh = np.arctan2(np.sin(H[-1, 2] - H[-5, 2]), np.cos(H[-1, 2] - H[-5, 2]))
        ego = np.r_[[spd, v_prev, float(dh) / 0.5], hist.reshape(-1)].astype(np.float32)

        ag = np.zeros((32, 28), np.float32)
        am = np.zeros(32, np.float32)
        rows = []
        for a in self._vehicle.get_world().get_actors().filter('vehicle.*'):
            if a.id == self._vehicle.id:
                continue
            t = a.get_transform()
            p = np.array([t.location.x, t.location.y])
            rows.append((float(np.linalg.norm(p - np.array([x, y]))), p, t, a))
        rows.sort(key=lambda r: r[0])
        for k, (_, p, t, a) in enumerate(rows[:32]):
            dd = p - np.array([x, y])
            rel = np.array([dd[0] * c - dd[1] * s, dd[0] * s + dd[1] * c])
            dh2 = np.deg2rad(t.rotation.yaw) - h
            v = a.get_velocity()
            sp = float(np.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))
            bb = a.bounding_box.extent
            ag[k] = np.r_[[rel[0], rel[1], np.cos(dh2), np.sin(dh2), sp,
                           2 * bb.x, 2 * bb.y, 1.0], np.zeros(20, np.float32)]
            am[k] = 1.0

        town = self._vehicle.get_world().get_map().name.split('/')[-1].replace('_Opt', '')
        mp = np.zeros((128, 7), np.float32)
        mm = np.zeros(128, np.float32)
        got = self._map(town)
        if got is not None:
            tree, mid, slen, shd = got
            kq = min(400, len(mid))
            dist, idx = tree.query([x, y], k=kq)
            # 只保留 30 m 内、最近的 128 段
            keep = np.asarray(idx)[np.asarray(dist) <= 30.0][:128]
            if len(keep):
                dv = mid[keep] - np.array([x, y])
                ex = dv[:, 0] * c - dv[:, 1] * s
                ey = dv[:, 0] * s + dv[:, 1] * c
                hh = shd[keep] - h
                n = len(keep)
                mp[:n, 0] = ex
                mp[:n, 1] = ey
                mp[:n, 2] = slen[keep]
                mp[:n, 3] = np.cos(hh)
                mp[:n, 4] = np.sin(hh)
                mp[:n, 5] = 1.0
                mp[:n, 6] = 1.0
                mm[:n] = 1.0
        return ego, ag, am, mp, mm

    def _control(self, wp41, spd):
        # 横向：预瞄 + 近端航向
        idx = min(25, len(wp41) - 1)
        tx, ty = wp41[idx]
        alpha = np.arctan2(ty, max(tx, 0.5))
        hd = np.arctan2(wp41[5][1], max(wp41[5][0], 0.5))
        steer = float(np.clip(1.0 * alpha + 0.5 * hd, -0.6, 0.6))

        # 纵向：几何限速（曲率）+ 计划速度，取小
        k = 12
        seg = np.diff(wp41[:min(len(wp41), 41)], axis=0)
        th = np.unwrap(np.arctan2(seg[:, 1], seg[:, 0]))
        curv = np.abs(np.gradient(th)) / np.maximum(
            np.linalg.norm(seg, axis=1), 1e-3)
        kappa = float(np.median(curv[:k])) if len(curv) else 0.0
        v_geo = float(np.sqrt(2.0 / max(kappa, 1e-4)))      # a_lat = 2 m/s^2
        n = min(len(wp41), 21)
        v_plan = float(np.mean(np.linalg.norm(np.diff(wp41[:n], axis=0), axis=1)) * 10.0)
        tgt = float(np.clip(min(v_geo, max(v_plan, 3.0)), 3.0, 14.0))

        # 卡死检测：慢速但一直给油 → 刹停一下再试
        if spd < 0.8 and self._last_thr > 0.5:
            self._stuck_t += 0.05
        else:
            self._stuck_t = max(0.0, self._stuck_t - 0.1)
        if self._stuck_t > 1.5:
            self._spd_i = 0.0
            return steer, 0.0, 1.0

        # PI 速度控制
        err = tgt - spd
        self._spd_i = float(np.clip(self._spd_i + err * 0.05, -2.0, 6.0))
        acc = 0.45 * err + 0.25 * self._spd_i
        thr = float(np.clip(acc, 0.0, 0.8))
        brk = float(np.clip(-1.2 * err, 0.0, 1.0)) if err < -0.5 else 0.0
        return steer, thr, brk

    def destroy(self, results=None):
        try:
            if self.dbg:
                np.savez_compressed('/mnt/2T_HDD/plant2_vector/dbg.npz',
                    route=np.stack([d['route'] for d in self.dbg]),
                    pred=np.stack([d['pred'] for d in self.dbg]),
                    pose=np.stack([d['pose'] for d in self.dbg]),
                    k=np.array([d['k'] for d in self.dbg]),
                    ctrl=np.array([[d['steer'], d['thr'], d['brk']] for d in self.dbg]),
                    modes=np.stack([d['modes'] if d.get('modes') is not None else np.zeros((6, 8, 2), np.float32) for d in self.dbg]),
                    mp=np.stack([d['mp'] for d in self.dbg]),
                    mm=np.stack([d['mm'] for d in self.dbg]))
                print('[VecAgent] dbg saved', len(self.dbg), flush=True)
        except Exception as e:
            print('[VecAgent] dbg save fail', e, flush=True)
        super().destroy()

    @torch.inference_mode()
    def run_step(self, input_data, timestamp):
        if self._vehicle is None:
            self._vehicle = self._find_hero()
        pose = self._pose()
        ego, ag, am, mp, mm = self._feat(pose)
        if self._rp is None or self.step % 5 == 0:
            self._rp = self._route(pose)
        rp = self._rp
        rp7 = np.concatenate([rp[:, :2], np.full((24, 1), 2.0, np.float32), rp[:, 3:4],
                              np.sqrt(np.maximum(1 - rp[:, 3:4] ** 2, 0)),
                              np.full((24, 1), 2.0, np.float32),
                              (np.abs(rp[:, :1]) + np.abs(rp[:, 1:2]) > 0).astype(np.float32)], 1)
        MP = np.concatenate([mp, rp7], 0)
        MM = np.concatenate([mm, rp7[:, 6]])
        b = [torch.tensor(x[None], device=self.device) for x in (ego, ag, am, MP, MM)]
        if ROUTE_ONLY:
            # 基线：绕开模型，沿 route 中心线以固定速度取 8 点
            rp_v = rp[np.abs(rp[:, 0]) + np.abs(rp[:, 1]) > 0]
            v_tgt = 6.0
            ss = v_tgt * 0.5 * np.arange(1, 9)
            if len(rp_v) >= 3:
                s_max = float(rp_v[-1, 2])
                ss = np.clip(ss, 0.0, s_max)
                wp = np.stack([np.interp(ss, rp_v[:, 2], rp_v[:, 0]),
                               np.interp(ss, rp_v[:, 2], rp_v[:, 1])], 1)
                # 末端不足的部分沿 route 末方向延伸
                if s_max < v_tgt * 4.0:
                    d = rp_v[-1, :2] - rp_v[max(len(rp_v) - 2, 0), :2]
                    d = d / max(np.linalg.norm(d), 1e-6)
                    extra = (v_tgt * 0.5 * np.arange(1, 9) - s_max)[:, None]
                    wp = wp + np.clip(extra, 0, None) * d[None]
            else:
                wp = np.stack([v_tgt * 0.5 * np.arange(1, 9), np.zeros(8)], 1)
            k = 0
        else:
            pred, logits = self.model(*b)
            k = int(logits.argmax(1)[0])
            wp_all = pred[0].detach().cpu().numpy()
            wp = wp_all[k]
        # ---- 2 Hz 重规划 + 20 Hz 跟踪（计划存世界坐标，每帧转回 ego 系）----
        if self.step % REPLAN_EVERY == 0 or self._plan_world is None:
            x0, y0, h0, _ = pose
            c0, s0 = np.cos(h0), np.sin(h0)
            Pw = np.vstack([[0.0, 0.0], wp])
            self._plan_world = np.stack(
                [x0 + Pw[:, 0] * c0 - Pw[:, 1] * s0,
                 y0 + Pw[:, 0] * s0 + Pw[:, 1] * c0], 1)
            self._plan_mode = k
        x, y, h, _ = pose
        c, s = np.cos(-h), np.sin(-h)
        d = self._plan_world - np.array([x, y])
        pe = np.stack([d[:, 0] * c - d[:, 1] * s, d[:, 0] * s + d[:, 1] * c], 1)
        tt_all = 0.5 * np.arange(len(pe))
        ahead = np.where(pe[:, 0] > 0.3)[0]
        j = int(ahead[0]) if len(ahead) else 0
        pe2, t2 = pe[j:], tt_all[j:]
        if len(pe2) < 3:
            pe2 = np.array([[0.0, 0.0], [8.0, 0.0], [16.0, 0.0]])
            t2 = np.array([0.0, 1.0, 2.0])
        t2 = t2 - t2[0]
        cs = CubicSpline(t2, pe2, axis=0)
        wp41 = cs(np.linspace(0.0, float(t2[-1]), 41))
        steer, thr, brk = self._control(wp41, pose[3])

        if DEBUG:
            _modes = wp_all if 'wp_all' in dir() else None
            self.dbg.append(dict(route=rp.copy(), pred=wp.copy(), modes=_modes,
                                pose=np.array(pose, np.float64), mp=mp.copy(), mm=mm.copy(),
                                steer=steer, thr=thr, brk=brk, k=k))
        self.step += 1
        if DEBUG and self.step % 20 == 0:
            _lg = 0.0 if ROUTE_ONLY else float(logits.max())
            print(f'[VecAgent] step {self.step} v={pose[3]:.1f} mode={k} '
                  f'logit={_lg:.2f} steer={steer:.2f} '
                  f'thr={thr:.2f} brk={brk:.2f}', flush=True)
        if REC and 'rgb' in input_data:
            try:
                import cv2
                img = input_data['rgb'][1][:, :, :3].copy()
                txt = 'v=%.1f thr=%.2f brk=%.2f steer=%.2f mode=%d' % (
                    float(self._pose()[3]), thr, brk, steer, k)
                cv2.putText(img, txt, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(img, txt, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imwrite(os.path.join(REC_DIR, 'frame_%05d.png' % self.step),
                            img[:, :, ::-1])
            except Exception as e:
                if self.step % 20 == 0:
                    print('rec fail', e, flush=True)

        self._last_thr = thr
        ctrl = carla.VehicleControl()
        ctrl.steer, ctrl.throttle, ctrl.brake = steer, thr, brk
        return ctrl
