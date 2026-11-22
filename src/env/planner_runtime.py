"""在线 PlannerRuntime：从 env 观测张量组装 planner 输入，按固定频率滚动重规划。

设计要点（与 IDC 训练口径对齐）：
  - 输入全部来自 env（absolute_self / self / partner_observations / map_observation），
    不再读 JSON、不再用日志帧；
  - 输出在 env 局部坐标，install_external 时用 target_pos 重锚定到自车当前位姿；
  - 参考速度用 env 内部同一口径（恒定 v_ref，默认 20 m/s），避免喂分布外输入；
  - 每 replan_every 步重规划一次，其余步沿用上一条参考。

两种模式：
  route   : 不调用模型，把 env 内部参考原样重装一遍（验收用，RC 应复现基线）
  planner : 模型出 K 条模态，取前 k 条注入
"""
import os
import sys

import numpy as np
import torch

# 让 runtime 能 import tools/train_planner.py（Planner 定义在那里）
_TOOLS = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'tools'))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

N_ENV = 91
T_FUT = 8          # 8 个 waypoint
HIST = 10          # 10 帧历史
AG_MAX = 32
MAP_MAX = 128
SPEED_REF = 20.0
# 环境坐标 y 轴方向与训练数据相反时，用 RT_MIRROR_Y=1 做镜像
MIRROR = float(os.environ.get('RT_MIRROR_Y', '1.0'))


class PlannerRuntime:
    def __init__(self, ckpt, k=1, replan_every=5, device='cuda',
                 mode='planner', speed_ref=SPEED_REF, history=HIST,
                 dump_path=''):
        from train_planner import Planner
        ck = torch.load(ckpt, map_location='cpu', weights_only=False)
        self.K = int(ck.get('K', 6))
        self.device = device
        self.model = Planner(K=self.K).to(device).eval()
        self.model.load_state_dict(ck['model'], strict=False)
        self.k = min(int(k), self.K)
        self.replan_every = int(replan_every)
        self.mode = mode
        self.speed_ref = float(speed_ref)
        self.H = int(history)
        self.buf = {}          # world -> list of (x, y, heading, speed)
        self.last = {}         # world -> 上一次的规划结果（env 坐标 91 点）
        self.n_replan = 0
        self.n_nan = 0
        self.dump_path = dump_path
        self.debug = False
        self.n_jump = 0
        self.dead = set()
        self.step = 0
        self.orig = {}         # world -> env 自己的 91 点参考（第一次重规划时抓）
        self.packs = {}        # world -> ScenePack（P0：离线同一套特征）
        self.dumps = {}        # world -> dict(steps=[], cands=[], sel=[], tgt=[], ego=[])

    # ---------------- 观测缓冲 ----------------
    def reset(self):
        self.buf = {}
        self.orig = {}
        self.last = {}
        self.n_replan = 0
        self.dead = set()
        self.step = 0
        self.dumps = {}

    def observe(self, env, ego_indices):
        """每步调用：记录自车位姿/速度（world/env 坐标）。"""
        abs_np = env.sim.absolute_self_observation_tensor().to_torch().cpu().numpy()
        rel_np = env.sim.self_observation_tensor().to_torch().cpu().numpy()
        self.step += 1
        for w, a in enumerate(ego_indices):
            x, y = float(abs_np[w, a, 0]), float(abs_np[w, a, 1])
            h = float(abs_np[w, a, 7])
            v = float(rel_np[w, a, 0])
            # GPUDrive 对失效车辆返回 -11000 之类的哨兵值：这类 world 不能重规划
            if (not np.isfinite([x, y, h]).all()) or abs(x) > 5000 or abs(y) > 5000:
                if w not in self.dead:
                    self.dead.add(w)
                    _prev = (self.buf.get(w) or [(float('nan'),) * 2])[-1]
                    try:
                        info = env.sim.info_tensor().to_torch().cpu().numpy()
                        _off, _col = float(info[w, a, 0]), float(info[w, a, 1])
                    except Exception:
                        _off = _col = float('nan')
                    print(f'[RT] world{w} 失效 @第{self.step}步 上次位姿='
                          f'({_prev[0]:.1f},{_prev[1]:.1f}) 新位姿=({x:.0f},{y:.0f}) '
                          f'offroad={_off:.0f} collision={_col:.0f}', flush=True)
                self.buf[w] = []
                continue
            self.dead.discard(w)
            d = self.buf.setdefault(w, [])
            # 位姿突变/世界重排 → 视为新一轮，清空历史（否则历史帧来自别的场景）
            if d and (abs(x - d[-1][0]) > 20.0 or abs(y - d[-1][1]) > 20.0):
                if self.debug and self.n_jump <= 5:
                    print(f'[RT] world{w} 位姿突变 '
                          f'({d[-1][0]:.0f},{d[-1][1]:.0f})→({x:.0f},{y:.0f})，清空历史',
                          flush=True)
                self.n_jump += 1
                d = []
                self.buf[w] = d
            d.append((x, y, h, v))
            if len(d) > self.H:
                d.pop(0)
            if self.dump_path:
                _dm = self.dumps.setdefault(w, dict(steps=[], cands=[], sel=[],
                                                    tgt=[], ego=[]))
                _dm['ego'].append((x, y))
                self._save_dump()

    def _ensure_pack(self, env, w):
        """惰性加载场景 JSON，用离线同一套特征实现（P0）。"""
        if w in self.packs:
            return self.packs[w]
        pk = None
        try:
            from env.planner_features import load_pack
            idx = list(env.data_loader.indices)[w]
            pk = load_pack(str(env.data_loader.dataset[idx]))
        except Exception as _e:
            print(f'[RT] world{w} ScenePack 加载失败: {_e}', flush=True)
        self.packs[w] = pk
        return pk

    # ---------------- 特征 ----------------
    def _ego_feat(self, w):
        d = self.buf[w]
        x, y, h, v = d[-1]
        hist = np.zeros((self.H, 2), np.float32)
        for i, (px, py, _, _) in enumerate(d[-self.H:]):
            dx, dy = px - x, py - y
            c, s = np.cos(h), np.sin(h)
            hist[i] = [dx * c - dy * s, MIRROR * (dx * s + dy * c)]
        pad = self.H - len(hist)
        if pad > 0:
            hist = np.vstack([np.repeat(hist[:1], pad, 0), hist])
        # 速度与偏航角速度：用缓冲里的历史差分（对齐训练口径：t 与 t-2）
        k = min(2, len(d) - 1) if len(d) > 1 else 1
        v_now = float(np.linalg.norm(np.array(d[-1][:2]) - np.array(d[-1 - k][:2])) / (k * 0.1)) \
            if len(d) > 1 else v
        v_prev = v_now
        if len(d) > 5:
            j = -5
            k2 = min(2, len(d) + j - 1) if (len(d) + j - 1) >= 0 else 1
            v_prev = float(np.linalg.norm(np.array(d[j][:2]) - np.array(d[j - k2][:2])) / (k2 * 0.1))
        yaw = float(np.arctan2(np.sin(h - d[max(0, len(d) - 6)][2]),
                               np.cos(h - d[max(0, len(d) - 6)][2]))) / max(0.1 * min(5, len(d) - 1), 1e-3)
        ego = np.concatenate([[v_now, v_prev, yaw], hist.reshape(-1)]).astype(np.float32)
        return np.nan_to_num(ego[:23], nan=0.0, posinf=0.0, neginf=0.0), (x, y, h)

    def _agent_feat(self, env, w, a, x, y, h):
        partner = env.sim.partner_observations_tensor().to_torch().cpu().numpy()[w, a]
        ag = np.zeros((AG_MAX, 28), np.float32)
        am = np.zeros(AG_MAX, np.float32)
        rows = []
        for p in partner:
            sp, rx, ry, rh = float(p[0]), float(p[1]), float(p[2]), float(p[3])
            if sp == 0.0 and rx == 0.0 and ry == 0.0:
                continue
            if not np.isfinite([sp, rx, ry, rh]).all():
                continue
            if abs(rx) > 1e3 or abs(ry) > 1e3:
                continue
            rows.append((np.hypot(rx, ry), sp, rx, ry, rh))
        rows.sort(key=lambda t: t[0])
        for n, (_, sp, rx, ry, rh) in enumerate(rows[:AG_MAX]):
            ag[n, :8] = [rx, ry, np.cos(rh), np.sin(rh), sp, 4.5, 1.9, 1.0]
            am[n] = 1.0
        return ag, am

    def _map_feat(self, env, w, x, y, h):
        road = env.sim.map_observation_tensor().to_torch().cpu().numpy()[w]
        mp = np.zeros((MAP_MAX, 7), np.float32)
        mm = np.zeros(MAP_MAX, np.float32)
        if road is None or len(road) == 0:
            return mp, mm
        segs = []
        for typ, is_lane in ((3, 1.0), (1, 0.0)):
            pts = road[road[:, 6] == typ]
            if len(pts) < 2:
                continue
            A, B = pts[:-1, :2], pts[1:, :2]
            mid = 0.5 * (A + B)
            ln = np.linalg.norm(B - A, axis=1)
            ok = (ln > 1e-3) & (ln < 20.0)
            for i in np.nonzero(ok)[0]:
                segs.append((np.hypot(mid[i, 0] - x, mid[i, 1] - y), mid[i], ln[i],
                             np.arctan2(B[i, 1] - A[i, 1], B[i, 0] - A[i, 0]), is_lane))
        segs.sort(key=lambda t: t[0])
        for n, (dd, mid, ln, ang, is_lane) in enumerate(segs[:MAP_MAX]):
            dx, dy = mid[0] - x, mid[1] - y
            c, s = np.cos(h), np.sin(h)
            mp[n] = [dx * c - dy * s, MIRROR * (dx * s + dy * c), ln,
                     np.cos(ang - h), MIRROR * np.sin(ang - h), is_lane, 1.0]
            mm[n] = 1.0
        return mp, mm

    # ---------------- 重规划 ----------------
    def maybe_replan(self, step, env, builder, ego_indices, install_fn):
        if step % self.replan_every != 0:
            return None
        # 抓一份 env 自己的 91 点参考，后面用来补尾段（避免直线外推）
        for _w in range(builder.num_worlds):
            if _w in self.orig or _w in self.dead:
                continue
            try:
                _a0 = int(ego_indices[_w])
                _o = builder.candidate_paths[_w][_a0][0]['pos']
                self.orig[_w] = np.asarray(_o, np.float32).copy()
            except Exception:
                pass
        paths, speeds, tgt = {}, {}, {}
        if self.mode == 'route':
            # route 的参考是完整路线（RC 按它算），只在第 0 步装一次，不做中途重锚定
            if step > 0:
                return None
            for w in range(builder.num_worlds):
                if w in self.dead:
                    continue
                o = self.orig.get(w)
                x, y = self.buf[w][-1][0], self.buf[w][-1][1]
                if o is None:
                    a = int(ego_indices[w])
                    cp = builder.candidate_paths[w][a][0]
                    paths[w] = np.asarray(cp['pos'], np.float32)[None]
                    speeds[w] = np.asarray(cp['speed'], np.float32)[None]
                    tgt[w] = np.asarray(cp['pos'][0], np.float32)
                    continue
                lo = int(min(max(step, 0), max(len(o) - 41, 0)))
                if getattr(self, 'debug', False) and self.n_replan < 20 and w == 0:
                    _d = np.linalg.norm(np.asarray(o, np.float32) - np.array([x, y], np.float32), axis=1)
                    print('[RT][slice] step=%d lo=%d 最近点=%d 距离=%.2f o[lo]=(%.1f,%.1f) ego=(%.1f,%.1f)'
                          % (step, lo, int(np.argmin(_d)), float(_d.min()),
                             float(o[lo][0]), float(o[lo][1]), x, y), flush=True)
                seg = np.asarray(o[lo:lo + 41], np.float32).copy()
                if len(seg) < 2:
                    continue
                full = np.asarray(o, np.float32) - np.asarray(o[lo], np.float32) + np.array([x, y], np.float32)
                paths[w] = full[None]
                speeds[w] = np.full((1, N_ENV), self.speed_ref, np.float32)
                tgt[w] = np.array([x, y], np.float32)
        else:
            for w in range(builder.num_worlds):
                if w in self.dead:
                    continue
                a = int(ego_indices[w])
                ego, (x, y, h) = self._ego_feat(w)
                pk = self._ensure_pack(env, w)
                if pk is not None:
                    _t = min(10 + int(step), 90)
                    h = -h            # env 的 yaw 与日志反向
                    ego = pk.ego_feat(_t)
                    ag, am = pk.agent_feat(_t, (x, y, h))
                    mp, mm = pk.map_feat((x, y, h))
                else:
                    ag, am = self._agent_feat(env, w, a, x, y, h)
                    mp, mm = self._map_feat(env, w, x, y, h)
                if self.debug and self.n_replan < 8:
                    print('[RT][feat] step=%d w=%d pose=(%.1f,%.1f,%.2f) '
                          'ego[absmax=%.1f] ag[absmax=%.1f n=%d] map[absmax=%.1f n=%d]'
                          % (step, w, x, y, h,
                             float(np.abs(ego).max()), float(np.abs(ag).max()),
                             int(am.sum()), float(np.abs(mp).max()), int(mm.sum())),
                          flush=True)
                with torch.no_grad():
                    pred = self.model(torch.tensor(ego, device=self.device)[None],
                                      torch.tensor(ag, device=self.device)[None],
                                      torch.tensor(am, device=self.device)[None],
                                      torch.tensor(mp, device=self.device)[None],
                                      torch.tensor(mm, device=self.device)[None])[0]
                pred = pred.cpu().numpy()[:self.k]                 # (k,8,2) ego frame
                if not np.isfinite(pred).all():
                    self.n_nan += 1
                    if self.n_nan <= 3:
                        print(f'[RT] world{w} 模型输出含 NaN/Inf，跳过本次重规划 '
                              f'(ego有限={np.isfinite(ego).all()} '
                              f'ag有限={np.isfinite(ag).all()} '
                              f'mp有限={np.isfinite(mp).all()})', flush=True)
                    continue
                pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
                P = []
                for kk in range(pred.shape[0]):
                    p40 = np.vstack([[0.0, 0.0], self._densify(pred[kk], 40)])
                    c, s = np.cos(-h), np.sin(-h)
                    P.append(np.stack([x + p40[:, 0] * c - p40[:, 1] * s,
                                       y + MIRROR * (p40[:, 0] * s + p40[:, 1] * c)], 1))
                P = np.stack(P)
                ext = np.stack([self._complete(p, self.orig.get(w)) for p in P])
                paths[w] = ext.astype(np.float32)
                speeds[w] = np.full((len(P), N_ENV), self.speed_ref, np.float32)
                tgt[w] = np.array([x, y], np.float32)
        install_fn(builder, ego_indices, paths, speeds, tgt)
        if self.dump_path:
            for _w in paths:
                if _w in self.dead:
                    continue
                _dm = self.dumps.setdefault(_w, dict(steps=[], cands=[], sel=[],
                                                     tgt=[], ego=[]))
                _dm['steps'].append(int(step))
                _dm['cands'].append(np.asarray(paths[_w], np.float32))
                _dm['sel'].append(-1)         # 由调用方回填
                _dm['tgt'].append(np.asarray(tgt[_w], np.float32))
            self._save_dump()
        self.n_replan += 1
        return paths

    def _save_dump(self):
        if not self.dump_path:
            return
        out = {}
        for w, dm in self.dumps.items():
            out[f'w{w}_steps'] = np.asarray(dm['steps'], np.int32)
            out[f'w{w}_cands'] = (np.stack(dm['cands']) if dm['cands']
                                  else np.zeros((0, 1, N_ENV, 2), np.float32))
            out[f'w{w}_sel'] = np.asarray(dm['sel'], np.int32)
            out[f'w{w}_tgt'] = (np.stack(dm['tgt']) if dm['tgt']
                                else np.zeros((0, 2), np.float32))
            out[f'w{w}_ego'] = np.asarray(dm['ego'], np.float32)
        try:
            np.savez_compressed(self.dump_path, **out)
        except Exception:
            pass

    def set_selected(self, idx):
        """记录本次重规划被选中的候选下标（用于可视化）。"""
        if not self.dump_path:
            return
        for dm in self.dumps.values():
            if dm['sel']:
                dm['sel'][-1] = int(idx)
        self._save_dump()

    # ---------------- 工具 ----------------
    @staticmethod
    def _densify(p8, n=40):
        t_src = np.linspace(0, 4.0, len(p8))
        t_dst = np.linspace(0, 4.0, n)
        return np.stack([np.interp(t_dst, t_src, p8[:, 0]),
                         np.interp(t_dst, t_src, p8[:, 1])], 1)

    @staticmethod
    def _complete(p41, orig, n=N_ENV):
        """前 4 s 用规划，之后接回 env 自己的路线（而不是直线外推）。"""
        p41 = np.asarray(p41, np.float32)
        if orig is not None and len(orig) >= n and len(p41) >= 41:
            tail = np.asarray(orig[41:n], np.float32) - np.asarray(orig[41], np.float32) + p41[40]
            return np.vstack([p41[:41], tail])
        return PlannerRuntime._extend(p41, n)

    @staticmethod
    def _extend(p, n=N_ENV):
        if len(p) >= n:
            return p[:n]
        seg = np.linalg.norm(np.diff(p[-11:], axis=0), axis=1) / 0.1
        v = float(np.clip(np.median(seg[seg > 0.3]) if np.any(seg > 0.3) else 5.0,
                          1.0, 20.0))
        d = p[-1] - p[-2]
        hh = float(np.arctan2(d[1], d[0])) if np.linalg.norm(d) > 1e-6 else 0.0
        step = np.array([np.cos(hh), np.sin(hh)]) * v * 0.1
        out = list(p)
        q = p[-1].copy()
        while len(out) < n:
            q = q + step
            out.append(q.copy())
        return np.asarray(out, np.float32)














