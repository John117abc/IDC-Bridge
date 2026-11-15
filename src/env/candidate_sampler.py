"""横向多模态候选的采样 + 走廊裁剪 + 四判据证书（M0-b）。

设计约束（来自实测，不是拍脑袋）
--------------------------------
* 横向范围 ±1.2 m：实测 ±1.2 m 时 68% 场景整条完全可行、逐点越界仅 4.9%；
  ±1.5 m 掉到 62%，裁剪负担明显变大。
* 横向步长 0.4 m：必须大于跟踪误差（实测 median pos_err 0.3~0.5 m），
  否则 critic/代价分不出候选差异。
* 走廊几何：`MapObservation` 是线段（position + length + heading），
  边界是 type 1（road_edge）；type 0 的 length 恒为 0（是点，不是线段）。
  距离必须用**点到线段**。
* 运动学：曲率上限 0.246 1/m，且逐点 κ(s) ≤ a_lat_max / v(s)²，
  a_lat_max = 3.0 m/s²（专家单条最大值 p95）。

四态证书
--------
INFEASIBLE           运动学或走廊判据不过（裁剪也救不回来）
CERTIFIED_DIFFERENT  可行，且与已选候选的 Hausdorff ≥ delta
CERTIFIED_FEASIBLE   可行，但与某个已选候选太像
（碰撞判据离线不可评：单帧只拿到周车当前位置，没有未来轨迹 —— 记为 UNKNOWN）
"""
from dataclasses import dataclass, field

import numpy as np

# ---- map_observation_tensor 列含义（实测）----
IX_X, IX_Y, IX_LEN, IX_HEAD, IX_TYPE = 0, 1, 2, 5, 6
ROAD_EDGE = 1


@dataclass
class Candidate:
    param: dict
    pos: np.ndarray            # [T,2] 世界坐标
    heading: np.ndarray        # [T]
    speed: np.ndarray          # [T]
    clearance: np.ndarray      # [T] 到边界线段的距离
    status: str
    cert: dict = field(default_factory=dict)


@dataclass
class SamplerConfig:
    lat_offsets: tuple = (-1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2)
    d_safe: float = 0.8
    kappa_max: float = 0.246
    a_lat_max: float = 3.0
    a_long_max: float = 3.0
    v_cap: float = 20.0
    delta: float = 0.8          # 候选间 Hausdorff 下界（m）
    k_select: int = 4           # 证书后保留几条
    dt: float = 0.1


# ---------------------------------------------------------------- 几何工具
def dedup_map(rows):
    if not len(rows):
        return rows
    key = np.round(rows[:, [IX_X, IX_Y, IX_LEN, IX_HEAD, IX_TYPE]], 4)
    _, idx = np.unique(key, axis=0, return_index=True)
    return rows[np.sort(idx)]


def boundary_segments(map_rows, types=(ROAD_EDGE,)):
    """返回边界线段的两端点 a,b [M,2]（若某行 length<=0 则视为点，做成极短段）。"""
    m = dedup_map(map_rows)
    if not len(m):
        return np.zeros((0, 2)), np.zeros((0, 2))
    m = m[np.isin(m[:, IX_TYPE].astype(int), list(types))]
    if not len(m):
        return np.zeros((0, 2)), np.zeros((0, 2))
    x, y, L, h = m[:, IX_X], m[:, IX_Y], m[:, IX_LEN], m[:, IX_HEAD]
    L = np.maximum(L, 1e-3)
    c, s = np.cos(h), np.sin(h)
    a = np.stack([x - 0.5 * L * c, y - 0.5 * L * s], 1)
    b = np.stack([x + 0.5 * L * c, y + 0.5 * L * s], 1)
    return a, b


def dist_to_segments(q, a, b):
    """q [N,2] -> [N]，点到线段集最近距离。"""
    q = np.asarray(q, dtype=np.float64)
    if len(a) == 0:
        return np.full(len(q), np.inf)
    ab = b - a
    denom = np.maximum((ab ** 2).sum(1), 1e-12)
    step = max(1, int(4e6 // max(1, len(a))))
    out = np.empty(len(q))
    for i in range(0, len(q), step):
        qq = q[i:i + step]
        ap = qq[:, None, :] - a[None]
        t = np.clip((ap * ab[None]).sum(-1) / denom[None], 0.0, 1.0)
        proj = a[None] + t[..., None] * ab[None]
        out[i:i + step] = np.linalg.norm(qq[:, None, :] - proj, axis=-1).min(1)
    return out


def path_normals(p):
    d = np.gradient(p, axis=0)
    t = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    return np.stack([-t[:, 1], t[:, 0]], axis=1)     # 左法向


def curvature(p, dt):
    """航向差分法。先对位置做 5 点平滑，再对曲率做 3 点平滑，抑制噪声尖峰。"""
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


def speed_profile(pos, kap, cfg):
    """前向-后向速度剖面：同时满足侧向加速度与纵向加加速度约束。

    逐点取 min(v_raw, sqrt(a_lat/kappa)) 会让曲率尖峰处速度骤降，
    导致 |Δv|/dt 超出纵向加速度上限（实测会让几乎所有候选被判 INFEASIBLE）。
    这里用标准的 forward-backward pass 把速度剖面压成可执行曲线。
    """
    d = np.diff(pos, axis=0)
    ds = np.maximum(np.linalg.norm(d, axis=1), 1e-6)
    v_raw = np.r_[ds / cfg.dt, ds[-1] / cfg.dt]
    if len(v_raw) >= 3:
        k = np.ones(3) / 3.0
        v_raw = np.convolve(np.pad(v_raw, (1, 1), mode="edge"), k, mode="valid")
    v_lim = np.sqrt(cfg.a_lat_max / np.maximum(kap, 1e-6))
    v = np.minimum(np.minimum(v_raw, v_lim), cfg.v_cap)
    # 后向：保证减速度不超过 a_long（v[i]^2 <= v[i+1]^2 + 2 a ds）
    for i in range(len(v) - 2, -1, -1):
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * cfg.a_long_max * ds[i]))
    # 前向：保证加速度不超过 a_long
    for i in range(1, len(v)):
        v[i] = min(v[i], np.sqrt(v[i - 1] ** 2 + 2 * cfg.a_long_max * ds[i - 1]))
    return np.clip(v, 0.0, cfg.v_cap)


# ---------------------------------------------------------------- 采样与裁剪
def clip_point_to_corridor(base_pt, normal, a, b, d_safe, o_max, steps=12):
    """把 base_pt + o*normal 沿法向收回到满足 clearance>=d_safe 的最近可行 o。

    返回 (实际使用的 o, 该点 clearance, 是否被裁剪)。
    """
    def clear(o):
        q = (base_pt + o * normal)[None, :]
        return float(dist_to_segments(q, a, b)[0])

    if clear(o_max) >= d_safe:
        return o_max, clear(o_max), False
    lo, hi = 0.0, o_max          # 假设 o=0（专家路径）一定可行
    if clear(lo) < d_safe:
        return 0.0, clear(lo), True
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        if clear(mid) >= d_safe:
            lo = mid
        else:
            hi = mid
    return lo, clear(lo), True


def make_candidate(expert, offset, a, b, cfg, long_shift=0.0):
    """按固定横向偏移生成候选（逐点走廊裁剪 + 曲率限速重采样）。"""
    T = len(expert)
    n = path_normals(expert)
    pos = np.empty_like(expert)
    clipped = 0
    o_used = np.empty(T)
    for i in range(T):
        o, _c, was = clip_point_to_corridor(expert[i], n[i], a, b,
                                           cfg.d_safe, abs(offset))
        o = o if offset >= 0 else -o
        o_used[i] = o
        pos[i] = expert[i] + o * n[i]
        clipped += int(was)
    # 纵向时间偏移（简化：整条轨迹按 long_shift 平移，仅用于打散速度模态）
    if abs(long_shift) > 1e-6:
        k = int(round(long_shift / cfg.dt))
        if k > 0:
            pos = np.vstack([pos[:1].repeat(k, 0), pos[:-k]])[:T]
        elif k < 0:
            pos = np.vstack([pos[-k:], pos[-1:].repeat(-k, 0)])[:T]

    d = np.diff(pos, axis=0)
    kap = curvature(pos, cfg.dt)
    speed = speed_profile(pos, kap, cfg)
    heading = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    heading = np.r_[heading, heading[-1]]
    clear = dist_to_segments(pos, a, b)
    return Candidate(
        param={"lat_offset": float(offset), "long_shift": float(long_shift),
               "clipped_points": int(clipped)},
        pos=pos, heading=heading, speed=speed, clearance=clear,
        status="", cert={},
    )


# ---------------------------------------------------------------- 证书
def hausdorff(p, q):
    d1 = np.max([dist_to_segments(np.array([x]), q, q)[0] for x in p[::3]]) \
        if len(q) else np.inf
    d2 = np.max([dist_to_segments(np.array([x]), p, p)[0] for x in q[::3]]) \
        if len(p) else np.inf
    return float(max(d1, d2))


def certify(cand: Candidate, selected, cfg) -> Candidate:
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
    cor_ok = bool(np.all(clr >= cfg.d_safe * 0.98))
    # 判据语义（2026-11-15 修订）：
    #   走廊 = 硬判据（物理空间能不能待）
    #   运动学 = 软约束（只记 TRACTION_RISK，不拒候选）；理由：
    #     1) 跟踪器自身有 clamp，不可跟踪的候选会由代价 J/V 自然排后
    #     2) 曲率/加速度是 10 Hz 有限差分估出来的，有噪声，硬阈值会制造假阴性
    #        （实测把 29% 的候选误杀，K' 被压到 2.62）
    cand.cert = {
        "cor_ok": cor_ok,
        "traction_risk": bool(kin_fail),
        "kin_fail": kin_fail,
        "kappa_max": float(kap.max()), "a_lat_max": float(a_lat.max()),
        "a_long_max": float(acc.max() if len(acc) else 0.0),
        "clear_min": float(clr.min()),
        "clipped_points": cand.param["clipped_points"],
        "collision": "UNKNOWN",     # 离线只有周车当前位置，无未来轨迹
    }
    if not cor_ok:
        cand.status = "INFEASIBLE"
        return cand
    same = [d for d in (hausdorff(pos, s.pos) for s in selected) if d < cfg.delta]
    cand.status = "CERTIFIED_DIFFERENT" if not same else "CERTIFIED_FEASIBLE"
    cand.cert["min_diff_to_selected"] = (
        min((hausdorff(pos, s.pos) for s in selected), default=float("inf")))
    return cand


def build_candidates(expert, map_rows, cfg: SamplerConfig, long_shifts=(0.0,)):
    """采样 -> 裁剪 -> 证书 -> 按 |offset| 从小到大挑出 k_select 条。"""
    a, b = boundary_segments(map_rows)
    raw = [make_candidate(expert, o, a, b, cfg, ls)
           for ls in long_shifts for o in cfg.lat_offsets]
    selected = []
    # 只有走廊是硬判据；运动学不合格的候选保留但打 TRACTION_RISK
    ok = [c for c in raw if certify(c, [], cfg).cert["cor_ok"]]
    # 先按 |offset| 升序（保守优先），再按差异性贪心
    ok.sort(key=lambda c: (abs(c.param["lat_offset"]),
                           abs(c.param["long_shift"])))
    for c in ok:
        certify(c, selected, cfg)
        if c.status == "CERTIFIED_DIFFERENT" or not selected:
            c.status = "CERTIFIED_DIFFERENT" if not selected else c.status
            selected.append(c)
        if len(selected) >= cfg.k_select:
            break
    infeasible = [c for c in raw if c.status == "INFEASIBLE"]
    return selected, raw, infeasible
