"""B+ 方案：用地图局部切向场积分出前向路径，再叠横向偏移生成候选。

核心思想（为了彻底去掉"专家未来"这一特权信息）：
  候选 = 从**当前位姿**出发，沿**地图车道切向场**向前推演 9 秒的一族轨迹
  地图只用两件事：① 局部切向/曲率（合法输入）  ② 走廊可行性（合法输入）

与旧 `_generate_path_from_road_graph` 的区别：
  旧版"从最近道路元素开始随便挑下一个" → 会跳到远处的路、走成阶梯、自交。
  这里每步只前进 1 m，并且**只接受与自己航向对齐（<60°）且 6 m 内**的车道段，
  因此不会跑飞；找不到就沿当前航向直行（并有步数上限保护）。

用法：
    from env.map_candidates import generate_map_candidates
    cands = generate_map_candidates(pose, map_rows, offsets, cfg)
"""
import numpy as np

IX_X, IX_Y, IX_LEN, IX_HEAD, IX_TYPE = 0, 1, 2, 5, 6
LANE_TYPES = (3, 2)          # 3=lane_center, 2=road_line（回退）
ROAD_EDGE = 1

from .candidate_sampler import (SamplerConfig, boundary_segments, corridor_ok,
                               curvature, dist_to_segments, hausdorff,
                               smoothstep5, speed_profile)


class LaneField:
    """把车道段整理成"点 -> 最近可对齐段航向"的查询结构。"""

    def __init__(self, map_rows, types=LANE_TYPES):
        m = map_rows[np.abs(map_rows).sum(1) > 1e-6]
        seg = m[np.isin(m[:, IX_TYPE].astype(int), list(types))]
        if len(seg) < 2:      # 退回到所有非边界段
            seg = m[m[:, IX_TYPE].astype(int) != ROAD_EDGE]
        L = np.maximum(seg[:, IX_LEN], 1e-3) if len(seg) else np.zeros(0)
        h = seg[:, IX_HEAD] if len(seg) else np.zeros(0)
        c, s = np.cos(h), np.sin(h)
        self.a = np.stack([seg[:, IX_X] - 0.5 * L * c,
                           seg[:, IX_Y] - 0.5 * L * s], 1) if len(seg) else np.zeros((0, 2))
        self.b = np.stack([seg[:, IX_X] + 0.5 * L * c,
                           seg[:, IX_Y] + 0.5 * L * s], 1) if len(seg) else np.zeros((0, 2))
        self.h = h
        self.ab = self.b - self.a
        self.den = np.maximum((self.ab ** 2).sum(1), 1e-12)

    def __len__(self):
        return len(self.h)

    def nearest_aligned(self, q, th, radius=6.0, cos_min=np.cos(np.deg2rad(60))):
        """在半径内挑**最顺着当前航向**的车道段（距离只是次要因素）。

        旧版只比"谁最近"，在路口会把直行错跟到横穿车道上（实测 p90 偏差 60~70 m）。
        改成 score = |航向差| + 0.05·距离 → 默认直行优先，符合车道连续性先验。
        """
        if not len(self.h):
            return np.inf, None
        t = np.clip(((q - self.a) * self.ab).sum(1) / self.den, 0.0, 1.0)
        proj = self.a + t[:, None] * self.ab
        d = np.linalg.norm(q - proj, axis=1)
        dth = np.abs(np.arctan2(np.sin(self.h - th), np.cos(self.h - th)))
        ok = (d < radius) & (dth < np.arccos(cos_min))
        if not ok.any():
            return np.inf, None
        idx = np.where(ok)[0]
        score = dth[idx] + 0.05 * d[idx]
        j = idx[int(np.argmin(score))]
        return float(d[j]), float(self.h[j])


def integrate_path(pose, field: LaneField, n=91, ds=1.0,
                   max_turn=np.deg2rad(4.0), lookahead=3.0):
    """从 pose=(x,y,theta) 出发，沿车道切向场前进 n 步，每步 ds 米。"""
    x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
    pts = np.empty((n, 2))
    ths = np.empty(n)
    pts[0] = (x, y)
    ths[0] = th
    for i in range(1, n):
        q = np.array([x + lookahead * np.cos(th), y + lookahead * np.sin(th)])
        _d, head = field.nearest_aligned(q, th)
        if head is not None:
            dth = np.arctan2(np.sin(head - th), np.cos(head - th))
            th = th + float(np.clip(dth, -max_turn, max_turn))
        x += ds * np.cos(th)
        y += ds * np.sin(th)
        pts[i] = (x, y)
        ths[i] = th
    return pts, np.unwrap(ths)


def offset_along_normals(pts, o):
    """沿逐点法向叠加 smoothstep 横向剖面（起点不动、终点偏 o）。"""
    P = np.asarray(pts, dtype=np.float64)
    d = np.diff(P, axis=0)
    t = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    t = np.vstack([t, t[-1:]])
    nvec = np.stack([-t[:, 1], t[:, 0]], 1)
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(d, axis=1))]
    u = arc / max(arc[-1], 1e-6)
    return P + (o * smoothstep5(u))[:, None] * nvec


def generate_map_candidates(pose, map_rows, offsets, cfg: SamplerConfig,
                            speed0=8.0, n=91, ds=1.0):
    """返回 list[Candidate]，每条带 corridor 可行性。

    pose: (x, y, theta) 当前位姿（世界系）
    speed0: 当前车速（来自仿真观测，因果）
    """
    from .candidate_sampler import Candidate
    field = LaneField(map_rows)
    base, _ = integrate_path(pose, field, n=n, ds=ds)
    a_seg, b_seg = boundary_segments(map_rows)
    out = []
    for o in offsets:
        p = offset_along_normals(base, o) if abs(o) > 1e-9 else base.copy()
        kap = curvature(p, cfg.dt)
        # 速度剖面：曲率限速 + 加减速约束，**起点用当前车速**，
        # 之后允许按 a_long_max 加速（不能把上限钉在当前车速上，
        # 否则从静止起步的场景参考速度永远是 0，车会一直爬）。
        spd = speed_profile(p, kap, cfg, scale=1.0)
        spd[0] = min(spd[0], float(np.clip(speed0, 0.5, cfg.v_cap)))
        ds = np.linalg.norm(np.diff(p, axis=0), axis=1)
        for i in range(1, len(spd)):
            spd[i] = min(spd[i],
                         np.sqrt(spd[i - 1] ** 2 + 2 * cfg.a_long_max * ds[i - 1]))
        d = np.diff(p, axis=0)
        hd = np.r_[np.unwrap(np.arctan2(d[:, 1], d[:, 0])), 0.0]
        hd[-1] = hd[-2]
        clear = dist_to_segments(p, a_seg, b_seg)
        c = Candidate(param={"lat_offset": float(o), "dt_shift": 0.0,
                             "speed_scale": 1.0},
                      pos=p, heading=hd, speed=spd, clearance=clear,
                      status="", cert={})
        c.cert = {"cor_ok": corridor_ok(clear, cfg),
                  "clear_min": float(clear.min())}
        c.status = "CERTIFIED_FEASIBLE" if c.cert["cor_ok"] else "INFEASIBLE"
        out.append(c)
    return out, base


def install_map_candidates(builder, ego_indices, maps, poses, speeds,
                           cfg: SamplerConfig, k_select: int = 5):
    """用**位姿 + 地图**生成候选（不读取任何专家未来），筛选后装进 builder。

    poses[w]  = (x, y, heading)    来自仿真观测（因果）
    speeds[w] = 当前车速           来自仿真观测（因果）

    返回实际生效的 K（各 world 取最大值，缺的用它自己的中心候选补齐）。
    """
    offsets = list(cfg.lat_offsets)
    if cfg.allow_wide:
        offsets += list(cfg.lat_offsets_wide)
    per_world = {}
    for w in range(builder.num_worlds):
        a = int(ego_indices[w])
        cands, _base = generate_map_candidates(poses[w], maps[w], offsets, cfg,
                                               speed0=speeds[w])
        center = next((c for c in cands
                       if abs(c.param["lat_offset"]) < 1e-9), None)
        ok = [c for c in cands if c.cert["cor_ok"]]
        # 中心候选兜底：它是沿车道走的，理应可行；若被判不可行，保留它
        if center is not None and center not in ok:
            center.cert["cor_ok"] = True
            center.status = "CERTIFIED_FEASIBLE"
            ok.append(center)
        ok.sort(key=lambda c: (abs(c.param["lat_offset"]),
                               c.param["dt_shift"]))
        sel = []
        for c in ok:
            nd = min((hausdorff(c.pos, s.pos) for s in sel),
                     default=float("inf"))
            if nd >= cfg.delta:
                sel.append(c)
            if len(sel) >= k_select:
                break
        if not sel and center is not None:
            sel = [center]
        per_world[w] = sel

    K = max(len(v) for v in per_world.values()) if per_world else 1
    for w in range(builder.num_worlds):
        a = int(ego_indices[w])
        sel = per_world[w]
        paths = [{
            'pos': c.pos.astype(np.float32),
            'heading': c.heading.astype(np.float32),
            'speed': c.speed.astype(np.float32),
            'road_dist': np.maximum(c.clearance, 0.0).astype(np.float32),
        } for c in sel]
        while len(paths) < K:                 # 用最后一条补齐，保证 ref 维齐
            paths.append(dict(paths[-1]))
        builder.candidate_paths[w] = {a: paths}
    builder.num_candidate_paths = K
    builder.rebuild_paths_tensor()
    return K, per_world
