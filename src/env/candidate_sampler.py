"""横向多模态候选的采样 + 走廊裁剪 + 四判据证书（M0-b，v2）。

v2 修订（2026-11-16）
--------------------
v1 用"专家路径整体法向平移"（平行偏移曲线），起点与 ego 就差 o 米，
相当于给跟踪器一个阶跃指令 —— 这不是"多模态轨迹"的正确语义。

v2 改成**同起点、不同终点**：
    y_cand(t) = y_expert(t) + o · S(u(t))
    S(u) = 6u^5 - 15u^4 + 10u^3      （smoothstep5）
    u    = 沿路径的弧长进度 0→1
于是  S(0)=S'(0)=S''(0)=0, S(1)=1, S'(1)=S''(1)=0
  • 起点 = ego 当前位置（初始跟踪误差 0）
  • 终点横向偏 o，且终点处与专家路径平行（不走成斜的）

纵向模态按 2026-11-16 的结论：**最近点投影下 pos_err 与时间无关**，
所以"让行"不需要新路径，只改速度剖面：
    speed_cand = speed_expert × T/(T+Δt)        Δt ∈ {0, 0.5, 1.0, 1.5} s

参数依据
--------
横向 o：0 / ±0.8 / ±1.6（±3.0 可选，默认关）
  • 0.8 ≈ 数据净横移 p90 (0.92 m)
  • 1.6 ≈ p95 (1.73) ≈ 走廊可行上限 p50 (1.50 m)
  • 间距 0.8 m > 跟踪噪声底 (median pos_err 0.3~0.5 m)
  • ±3.0 ≈ p99 (2.99)，但只有 1.2% 数据、48% 场景可行 → 默认关
"""
from dataclasses import dataclass, field

import logging

import numpy as np

IX_X, IX_Y, IX_LEN, IX_HEAD, IX_TYPE = 0, 1, 2, 5, 6
logger = logging.getLogger(__name__)
ROAD_EDGE = 1


@dataclass
class Candidate:
    param: dict
    pos: np.ndarray
    heading: np.ndarray
    speed: np.ndarray
    clearance: np.ndarray
    status: str
    cert: dict = field(default_factory=dict)


@dataclass
class SamplerConfig:
    # 一条参数轴，多档幅度：
    #   0      = 中心
    #   ±0.8   = 数据净横移 p90 (0.92 m) —— 车道内常见避让
    #   ±1.6   = p95 (1.73) ≈ 走廊可行上限 p50 (1.50 m) —— 大幅避让
    #   ±3.5   = 一个车道宽（Waymo 车道约 3.5 m）—— **换道**
    # 车道中心线本身就是"偏移曲线"，所以 o=3.5 的曲线几何上就是相邻车道中心线。
    # 能不能用交给走廊证书按场景判断（窄路/单车道会自动只剩小幅度）。
    lat_offsets: tuple = (-3.5, -1.6, -0.8, 0.0, 0.8, 1.6, 3.5)
    lat_offsets_wide: tuple = ()               # 已并入 lat_offsets，保留字段兼容
    # 纵向尺度：参考长度 = v_nominal × s × 9.1 s（v_nominal 来自地图曲率）
    # 实测专家 9.1 s 行驶距离 p50=47 m / p75=67 m / p90=171 m，固定长度覆盖不了这个跨度
    length_scales: tuple = (0.5, 1.0, 1.5)
    # 更直接的做法：直接给若干**锚点长度**（米），由数据集里程分位数来定
    # 45 / 70 / 110 / 170 ≈ 专家里程的 p50 / p75 / p85 / p90
    # 增加"慢/停"档：真实场景里前车排队/让行需要 1~3 m/s，只有 45 m(5 m/s) 起会全撞
    anchor_lengths: tuple = (15.0, 25.0, 45.0, 70.0, 110.0, 170.0)
    v_nominal_min: float = 4.0
    v_nominal_cap: float = 15.0
    allow_wide: bool = False
    dt_shifts: tuple = (0.0, 0.5, 1.0, 1.5)    # 纵向：晚到时长的目标
    d_safe: float = 0.8
    kappa_max: float = 0.246
    a_lat_max: float = 3.0
    a_long_max: float = 3.0
    v_cap: float = 20.0
    delta: float = 0.8
    k_select: int = 4
    dt: float = 0.1
    # 走廊判据的鲁棒化（2026-11-16）：
    # 原来要求"整条路径每一点都 >= d_safe"，tensor 里的边界是稀疏线段云，
    # 一个点擦到就整条废掉 —— 实测 3/16 场景连中心候选都被判 INFEASIBLE。
    # 改为：允许不超过 corridor_bad_frac 比例的点低于 d_safe，
    #      但绝对下限 clear_abs_min 必须满足。
    corridor_bad_frac: float = 0.05
    clear_abs_min: float = 0.15


def corridor_ok(clear, cfg):
    """鲁棒的走廊判据。"""
    clear = np.asarray(clear, dtype=np.float64)
    if not len(clear):
        return False
    bad = float(np.mean(clear < cfg.d_safe))
    return bool(bad <= cfg.corridor_bad_frac
                and float(clear.min()) >= cfg.clear_abs_min)


# ---------------------------------------------------------------- 几何工具
def dedup_map(rows):
    if not len(rows):
        return rows
    key = np.round(rows[:, [IX_X, IX_Y, IX_LEN, IX_HEAD, IX_TYPE]], 4)
    _, idx = np.unique(key, axis=0, return_index=True)
    return rows[np.sort(idx)]


def boundary_segments(map_rows, types=(ROAD_EDGE,)):
    m = dedup_map(map_rows)
    if not len(m):
        return np.zeros((0, 2)), np.zeros((0, 2))
    m = m[np.isin(m[:, IX_TYPE].astype(int), list(types))]
    if not len(m):
        return np.zeros((0, 2)), np.zeros((0, 2))
    x, y, L, h = m[:, IX_X], m[:, IX_Y], m[:, IX_LEN], m[:, IX_HEAD]
    L = np.maximum(L, 1e-3)
    c, s = np.cos(h), np.sin(h)
    return (np.stack([x - 0.5 * L * c, y - 0.5 * L * s], 1),
            np.stack([x + 0.5 * L * c, y + 0.5 * L * s], 1))


def dist_to_segments(q, a, b):
    q = np.asarray(q, dtype=np.float64)
    if len(a) == 0:
        return np.full(len(q), np.inf)
    ab = b - a
    den = np.maximum((ab ** 2).sum(1), 1e-12)
    step = max(1, int(4e6 // max(1, len(a))))
    out = np.empty(len(q))
    for i in range(0, len(q), step):
        qq = q[i:i + step]
        ap = qq[:, None, :] - a[None]
        t = np.clip((ap * ab[None]).sum(-1) / den[None], 0.0, 1.0)
        proj = a[None] + t[..., None] * ab[None]
        out[i:i + step] = np.linalg.norm(qq[:, None, :] - proj, axis=-1).min(1)
    return out


def smoothstep5(u):
    u = np.clip(np.asarray(u, dtype=np.float64), 0.0, 1.0)
    return u ** 3 * (10.0 - 15.0 * u + 6.0 * u * u)


def curvature(p, dt):
    ps = p.copy()
    if len(ps) >= 5:
        k = np.ones(5) / 5.0
        for d in range(2):
            ps[:, d] = np.convolve(np.pad(ps[:, d], (2, 2), mode="edge"), k,
                                   mode="valid")
    d = np.diff(ps, axis=0)
    ds = np.linalg.norm(d, axis=1)
    hd = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    kap = np.zeros(len(p))
    ds_mid = 0.5 * (ds[1:] + ds[:-1])
    ok = ds_mid > 0.15
    kap[1:-1][ok] = np.diff(hd)[ok] / ds_mid[ok]
    if len(kap) >= 3:
        k = np.ones(3) / 3.0
        kap = np.convolve(np.pad(kap, (1, 1), mode="edge"), k, mode="valid")
    return np.abs(kap)


def speed_profile(pos, kap, cfg, scale=1.0):
    d = np.diff(pos, axis=0)
    ds = np.maximum(np.linalg.norm(d, axis=1), 1e-6)
    v_raw = np.r_[ds / cfg.dt, ds[-1] / cfg.dt]
    if len(v_raw) >= 3:
        k = np.ones(3) / 3.0
        v_raw = np.convolve(np.pad(v_raw, (1, 1), mode="edge"), k, mode="valid")
    v_lim = np.sqrt(cfg.a_lat_max / np.maximum(kap, 1e-6))
    v = np.minimum(np.minimum(v_raw, v_lim), cfg.v_cap) * scale
    for i in range(len(v) - 2, -1, -1):
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * cfg.a_long_max * ds[i]))
    for i in range(1, len(v)):
        v[i] = min(v[i], np.sqrt(v[i - 1] ** 2 + 2 * cfg.a_long_max * ds[i - 1]))
    return np.clip(v, 0.0, cfg.v_cap)


# ---------------------------------------------------------------- 采样
def make_candidate(expert, o, dt_shift, a_seg, b_seg, cfg):
    """同起点、不同终点：在自车系里给路径叠加一条 smoothstep 横向剖面。"""
    T = len(expert)
    p0 = expert[0]
    d0 = expert[1] - expert[0]
    h0 = float(np.arctan2(d0[1], d0[0]))
    c, s = np.cos(-h0), np.sin(-h0)
    R = np.array([[c, -s], [s, c]])                  # 世界 -> 自车系
    pe = (expert - p0) @ R.T                          # [T,2]，起点在原点

    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pe, axis=0), axis=1))]
    u = arc / max(arc[-1], 1e-6)
    pe2 = pe.copy()
    pe2[:, 1] = pe[:, 1] + o * smoothstep5(u)          # 起点不动、终点偏 o

    # pe = (p - p0) @ R.T  ->  逆变换 pe2 @ R + p0
    pos = pe2 @ R + p0

    scale = T * cfg.dt / (T * cfg.dt + max(dt_shift, 0.0))
    kap = curvature(pos, cfg.dt)
    speed = speed_profile(pos, kap, cfg, scale=scale)
    d = np.diff(pos, axis=0)
    heading = np.r_[np.unwrap(np.arctan2(d[:, 1], d[:, 0])), 0.0]
    heading[-1] = heading[-2] if len(heading) > 1 else heading[-1]
    clear = dist_to_segments(pos, a_seg, b_seg)
    return Candidate(
        param={"lat_offset": float(o), "dt_shift": float(dt_shift),
               "speed_scale": float(scale)},
        pos=pos, heading=heading, speed=speed, clearance=clear,
        status="", cert={},
    )


def hausdorff(p, q, stride=3):
    if not len(p) or not len(q):
        return np.inf
    d1 = max(float(dist_to_segments(np.array([x]), q, q)[0]) for x in p[::stride])
    d2 = max(float(dist_to_segments(np.array([x]), p, p)[0]) for x in q[::stride])
    return max(d1, d2)


def certify(cand, selected, cfg):
    pos, spd, clr = cand.pos, cand.speed, cand.clearance
    kap = curvature(pos, cfg.dt)
    a_lat = spd ** 2 * kap
    acc = np.abs(np.diff(spd)) / cfg.dt
    kin_fail = []
    if np.any(kap > cfg.kappa_max * 1.05):
        kin_fail.append("kappa")
    if np.any(a_lat > cfg.a_lat_max * 1.05):
        kin_fail.append("a_lat")
    if len(acc) and np.any(acc > cfg.a_long_max * 1.5):
        kin_fail.append("a_long")
    cor_ok = corridor_ok(clr, cfg)
    cand.cert = {
        "cor_ok": cor_ok, "traction_risk": bool(kin_fail), "kin_fail": kin_fail,
        "kappa_max": float(kap.max()), "a_lat_max": float(a_lat.max()),
        "a_long_max": float(acc.max() if len(acc) else 0.0),
        "clear_min": float(clr.min()),
        "collision": "UNKNOWN",
    }
    if not cor_ok:
        cand.status = "INFEASIBLE"
        return cand
    same = [d for d in (hausdorff(pos, s.pos) for s in selected) if d < cfg.delta]
    cand.status = "CERTIFIED_FEASIBLE" if same else "CERTIFIED_DIFFERENT"
    cand.cert["min_diff"] = min((hausdorff(pos, s.pos) for s in selected),
                                default=float("inf"))
    return cand


def install_candidates(builder, ego_indices, maps, cfg: SamplerConfig):
    """把采样出的候选注入 builder.candidate_paths，并重建 ref_tensor。

    返回实际生效的候选数 K。K 在不同 world 可能不同（走廊太窄的会被判
    INFEASIBLE），这里取最大值、缺的用该 world 的最后一个候选补齐，
    保证 ref_tensor 的 path 维是齐的。
    """
    for w in range(builder.num_worlds):
        a = int(ego_indices[w])
        expert = builder.expert_pos[w, a].detach().cpu().numpy()
        if not np.all(np.abs(expert[:, :2]) < 1e3):
            continue                      # 无效轨迹，交给原来的单路径逻辑
        a_seg, b_seg = boundary_segments(maps[w])
        selected, _raw, _inf = build_candidates(expert, maps[w], cfg)
        if not selected:                  # 兜底：至少给一条中心候选
            selected = [make_candidate(expert, 0.0, 0.0, a_seg, b_seg, cfg)]
        paths = []
        for c in selected:
            paths.append({
                'pos': c.pos.astype(np.float32),
                'heading': c.heading.astype(np.float32),
                'speed': c.speed.astype(np.float32),
                'road_dist': np.maximum(c.clearance, 0.0).astype(np.float32),
            })
        builder.candidate_paths[w] = {a: paths}

    # 补齐到统一的 K
    K = 1
    for d in builder.candidate_paths.values():
        for paths in d.values():
            K = max(K, len(paths))
    for w, d in builder.candidate_paths.items():
        for a, paths in d.items():
            while len(paths) < K:
                paths.append(dict(paths[-1]))     # 浅拷贝即可（只读）
    builder.num_candidate_paths = K
    if hasattr(builder, 'rebuild_paths_tensor'):
        builder.rebuild_paths_tensor()
    return K


def install_fixed(builder, ego_indices, maps, offsets=(-0.8, 0.0, 0.8),
                  cfg: SamplerConfig = None):
    """A/B 对照用：给每个 world 固定安装 len(offsets) 条候选（不做证书筛选）。

    索引语义因此是确定的：第 i 条对应 offsets[i]（CenterIndex = len//2）。
    这让 eval 可以用 --force-path-index 精确指定"全部世界走哪条"。
    """
    cfg = cfg or SamplerConfig()
    for w in range(builder.num_worlds):
        a = int(ego_indices[w])
        expert = builder.expert_pos[w, a].detach().cpu().numpy()
        a_seg, b_seg = boundary_segments(maps[w])
        paths = []
        for o in offsets:
            c = make_candidate(expert, o, 0.0, a_seg, b_seg, cfg)
            paths.append({
                'pos': c.pos.astype(np.float32),
                'heading': c.heading.astype(np.float32),
                'speed': c.speed.astype(np.float32),
                'road_dist': np.maximum(c.clearance, 0.0).astype(np.float32),
            })
        builder.candidate_paths[w] = {a: paths}
    builder.num_candidate_paths = len(offsets)
    builder.rebuild_paths_tensor()
    return len(offsets)


def build_candidates(expert, map_rows, cfg: SamplerConfig):
    """采样 -> 证书 -> 贪心挑 k_select 条（优先 |o| 小、Δt 小）。"""
    a, b = boundary_segments(map_rows)
    offs = list(cfg.lat_offsets)
    if cfg.allow_wide:
        offs += list(cfg.lat_offsets_wide)
    raw = [make_candidate(expert, o, dt, a, b, cfg)
           for dt in cfg.dt_shifts for o in offs]
    ok = [c for c in raw if certify(c, [], cfg).cert["cor_ok"]]
    ok.sort(key=lambda c: (abs(c.param["lat_offset"]), c.param["dt_shift"]))
    selected = []
    for c in ok:
        certify(c, selected, cfg)
        if c.status == "CERTIFIED_DIFFERENT":
            selected.append(c)
        if len(selected) >= cfg.k_select:
            break
    infeasible = [c for c in raw if c.status == "INFEASIBLE"]
    return selected, raw, infeasible


def road_dist_for_path(pos, map_rows, last_valid_idx=None):
    """逐点到路缘的最近距离；与地图候选使用同一套定义与单位。

    pos: (N,2) env 坐标；last_valid_idx: 有效点个数（含），None=全部有效。
    返回 (dist (N,), has_edge bool)。无路缘时 dist 全为 inf，由调用方处理。
    """
    a_seg, b_seg = boundary_segments(map_rows)
    has_edge = len(a_seg) > 0
    if not has_edge:
        return np.full(len(pos), np.inf, np.float32), False
    d = dist_to_segments(np.asarray(pos, np.float64), a_seg, b_seg)
    return d.astype(np.float32), True


def install_external(builder, ego_indices, paths_by_world, speeds_by_world=None,
                    origin_pos=None, origin_heading=None, target_pos=None,
                    maps=None, last_valid_idx=None, blend_tail=False,
                    tail_idx=None):
    """把外部轨迹（如规划头的 top-k）直接注入为候选集。

    paths_by_world : {world: (K, N, 2)}  世界坐标
    speeds_by_world: {world: (K, N)}     可选，缺省按相邻点位移推
    blend_tail     : True 时，从 tail_idx 起的尾段沿用**原候选**（沿路线的那条），
                     避免 4 s 规划之后用直线外推导致脱轨
    tail_idx       : 从第几个点开始接原候选（默认取 last_valid_idx 对应的 nv）
    返回实际生效的 K。
    """
    K = 1
    for w in range(builder.num_worlds):
        a = int(ego_indices[w])
        P = paths_by_world.get(w)
        if P is None:
            continue
        # 外部轨迹是世界坐标，先转到 env 局部坐标（用 expert 首帧做刚体对齐）
        expert = builder.expert_pos[w, a].detach().cpu().numpy()
        env_p0 = expert[0, :2]
        d0 = expert[1, :2] - expert[0, :2]
        env_h0 = float(np.arctan2(d0[1], d0[0])) if np.linalg.norm(d0) > 1e-6 else 0.0
        lst = []
        _P = np.asarray(P, np.float32)
        if origin_pos is not None:
            jp0 = np.asarray(origin_pos[w], np.float32)
            jh0 = float(origin_heading[w])
            dth = env_h0 - jh0
            c, sn = np.cos(dth), np.sin(dth)
            R = np.array([[c, -sn], [sn, c]], np.float32)
            _P = ((_P.reshape(-1, 2) - jp0[None]) @ R.T + env_p0[None]).reshape(_P.shape)
        if target_pos is not None and w in target_pos:
            _tp = np.asarray(target_pos[w], np.float32)
            _delta = _tp - _P[0, 0]
            _P = _P + _delta[None, None, :]
        # 尾段接回原候选（沿路线），避免规划窗之后用直线外推脱轨
        _orig_pos = None
        if blend_tail:
            try:
                _orig = builder.candidate_paths[w][a][0]['pos']
                _orig_pos = np.asarray(_orig, np.float32)
                _ti = int(tail_idx) if tail_idx is not None else (
                    int(last_valid_idx[w]) if last_valid_idx is not None else len(_P[0]))
                _ti = int(np.clip(_ti, 1, _P.shape[1]))
                if len(_orig_pos) >= _P.shape[1]:
                    _delta_t = _P[:, _ti - 1] - _orig_pos[None, _ti - 1]
                    _tail = _orig_pos[None, _ti:] + _delta_t[:, None, :]
                    _P = np.concatenate([_P[:, :_ti], _tail], axis=1)
            except Exception as _e:
                logger.warning(f'[BLEND] world={w} 尾段接回失败: {_e}')
        # 路缘距离：与地图候选同一套定义（boundary_segments + dist_to_segments）
        _map_rows = None
        if maps is not None and w in maps:
            _map_rows = maps[w]
        elif getattr(builder, 'map_rows_by_world', None) and w in builder.map_rows_by_world:
            _map_rows = builder.map_rows_by_world[w]
        for i, p in enumerate(_P):
            d = np.gradient(p, axis=0)
            hd = np.arctan2(d[:, 1], d[:, 0]).astype(np.float32)
            if speeds_by_world is not None and w in speeds_by_world:
                sp = np.asarray(speeds_by_world[w][i], np.float32)
            else:
                sp = np.maximum(np.linalg.norm(np.diff(p, axis=0), axis=1) / 0.1,
                                0.5).astype(np.float32)
                sp = np.r_[sp, sp[-1]]
            nv = len(p) if last_valid_idx is None else int(last_valid_idx[w])
            nv = int(np.clip(nv, 1, len(p)))
            # 末点用向后差分（避免第一个外推点改变第 4 秒的速度/朝向特征）
            if nv >= 2 and nv < len(sp):
                sp[nv - 1] = sp[nv - 2]
                sp[nv:] = sp[nv - 1]
                hd[nv - 1] = np.float32(np.arctan2(p[nv - 1, 1] - p[nv - 2, 1],
                                                p[nv - 1, 0] - p[nv - 2, 0]))
                hd[nv:] = hd[nv - 1]
            if _map_rows is None:
                rd = np.zeros(len(p), np.float32)
            else:
                _d, _has = road_dist_for_path(p, _map_rows, nv)
                if not _has:
                    logger.warning(f'[ROAD_DIST] world={w} 无路缘线段，'
                                   f'road_dist 置 0（该场景缺路缘信息）')
                    rd = np.zeros(len(p), np.float32)
                else:
                    rd = np.asarray(_d, np.float32).copy()
                    if not np.isfinite(rd[:nv]).all():
                        logger.warning(f'[ROAD_DIST] world={w} 有效段含非有限值，已裁剪')
                        rd[:nv] = np.nan_to_num(rd[:nv], nan=0.0, posinf=50.0)
                    # 无效尾段：沿用有效末点，避免把外推段距离送进网络
                    if nv < len(p):
                        rd[nv:] = rd[nv - 1]
            lst.append(dict(pos=p, heading=hd, speed=sp, road_dist=rd,
                            last_valid_idx=int(nv) - 1))
        builder.candidate_paths[w] = {a: lst}
        K = max(K, len(lst))
    for w, d in builder.candidate_paths.items():
        for a, paths in d.items():
            while len(paths) < K:
                paths.append(dict(paths[-1]))
    builder.num_candidate_paths = K
    if hasattr(builder, 'rebuild_paths_tensor'):
        builder.rebuild_paths_tensor()
    return K
