"""IDC 66 维状态构造器（CARLA 版）

逐维口径与 GPUDrive 版 `src/env/idc_state_builder.py::get_idc_observations_batch`
一致（已核对源码）：

    ego       6  = [x, y, speed, 0, heading, 0]                 # 世界坐标、弧度
    others   32  = 8 辆最近车 × [gx, gy, abs_h, speed]           # 世界坐标
    validity  8  = 前 n_real 个为 1
    ref_err  18  = [dp, dphi, dv] + Σ_{t+5,10,15}[lat, dphi, road, spd]
                   + [curv_ahead, spd_ahead, road_ahead]
    temporal  1  = 参考投影索引 t（nearest 模式：±25 窗口内最近点、不倒退）
    veh       1  = 车长（训练里当轴距用，实测中位数 5.286）

参考路径需提供字段：pos(N,2), heading(N), speed(N), road_dist(N),
curv_ahead(N), spd_ahead(N), road_ahead(N), last_valid_idx(int, 可选)。
其中前四个由规划器 8 点密化 + 地图距离直接可得；
**curv_ahead / spd_ahead / road_ahead 的精确口径尚未从 GPUDrive 版抄全**
（见文件末尾 TODO），当前用近似，接入前需核对。
"""
import math

import numpy as np

DIM_EGO, DIM_OTHERS, DIM_VALIDITY, DIM_REF_ERROR = 6, 32, 8, 18
DIM_TEMPORAL, DIM_VEH = 1, 1
TOTAL_DIM = (DIM_EGO + DIM_OTHERS + DIM_VALIDITY + DIM_REF_ERROR
             + DIM_TEMPORAL + DIM_VEH)          # 66
N_OTHER = 8
REF_WINDOW = 25
DT = 0.1                                        # IDC 训练频率 10 Hz
HORIZON = 4.0                                   # 4 s 参考
N_REF = int(HORIZON / DT) + 1                   # 41 点
VEH_LEN_DEFAULT = 5.286


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def calc_ref_error(ego, ref):
    """[dp, dphi, dv]，dp 带符号（以参考航向为基准的横向偏移）。"""
    dx = float(ref[0] - ego[0])
    dy = float(ref[1] - ego[1])
    dp = math.hypot(dx, dy)
    cross = dy * math.cos(ref[4]) - dx * math.sin(ref[4])
    dp *= math.copysign(1.0, cross)
    dphi = wrap(float(ref[4] - ego[4]))
    dv = math.hypot(ego[2], ego[3]) - float(ref[2])
    return np.array([dp, dphi, dv], np.float32)


def project_index(pos, x, y, prev_idx, window=REF_WINDOW):
    """±window 内最近点投影，且索引不倒退（与 GPUDrive 版一致）。"""
    n = len(pos)
    prev = int(max(0, min(int(prev_idx), n - 1)))
    lo = max(0, prev - window)
    hi = min(n - 1, prev + window)
    seg = pos[lo:hi + 1]
    d = (seg[:, 0] - x) ** 2 + (seg[:, 1] - y) ** 2
    return max(prev, lo + int(np.argmin(d)))


def build_state(ego_pose, speed, others_world, path, ref_idx=0,
                veh_len=VEH_LEN_DEFAULT):
    """返回 (state[66], 新的 ref_idx)。

    ego_pose: (x, y, heading) 世界坐标 + 弧度
    others_world: [(gx, gy, abs_h, speed)]，已按距自车升序
    path: dict，字段见文件头
    """
    x, y, heading = float(ego_pose[0]), float(ego_pose[1]), float(ego_pose[2])
    ego = np.array([x, y, float(speed), 0.0, heading, 0.0], np.float32)

    o = np.zeros((N_OTHER, 4), np.float32)
    val = np.zeros(N_OTHER, np.float32)
    n = min(len(others_world), N_OTHER)
    for i in range(n):
        o[i] = np.asarray(others_world[i], np.float32)
        val[i] = 1.0

    pos = np.asarray(path['pos'], np.float32)
    t = project_index(pos, x, y, ref_idx)
    ref = np.array([pos[t, 0], pos[t, 1], path['speed'][min(t + 5, len(pos) - 1)], 0.0,
                    path['heading'][t], 0.0], np.float32)
    re = calc_ref_error(ego, ref)

    npts = len(pos)
    nv = int(path.get('last_valid_idx', npts - 1))
    nv = max(1, min(nv, npts - 1))
    la = []
    for dt in (5, 10, 15):
        k = min(t + dt, nv)
        lx, ly = float(pos[k, 0]), float(pos[k, 1])
        lh = float(path['heading'][k])
        lat = (ly - y) * math.cos(lh) - (lx - x) * math.sin(lh)
        la += [lat, wrap(lh - heading),
               float(path['road_dist'][k]), float(path['speed'][k])]
    la += [float(path['curv_ahead'][t]), float(path['spd_ahead'][t]),
           float(path['road_ahead'][t])]

    state = np.concatenate([
        ego,
        o.reshape(-1),
        val,
        re,
        np.asarray(la, np.float32),
        np.array([float(t)], np.float32),
        np.array([float(veh_len)], np.float32),
    ]).astype(np.float32)
    assert state.shape[0] == TOTAL_DIM, state.shape
    return state, t


# ----------------------------------------------------------------------
#  参考路径：规划器 8 点 (ego 系) → 4 s / 10 Hz 的 41 点 + 派生量
# ----------------------------------------------------------------------
def road_dist_of_points(pts_world, seg_a, seg_b, cap=200.0):
    """每个点到最近车道段的距离（cap 截断，与训练里 road_dist 量级一致）。"""
    pts = np.asarray(pts_world, np.float64)
    out = np.full(len(pts), cap, np.float32)
    if seg_a is None or len(seg_a) == 0:
        return out
    A = np.asarray(seg_a, np.float64)
    B = np.asarray(seg_b, np.float64)
    AB = B - A
    for i, p in enumerate(pts):
        tt = np.clip(np.sum((p - A) * AB, 1) / np.maximum(np.sum(AB * AB, 1), 1e-9), 0, 1)
        proj = A + tt[:, None] * AB
        out[i] = min(float(np.linalg.norm(p - proj, axis=1).min()), cap)
    return out


def build_path(plan_ego, pose, seg_a, seg_b, v_max=30.0):
    """规划器 8 点（ego 系，0.5 s 间隔）→ 41 点参考路径。

    返回 dict(pos, heading, speed, road_dist, curv_ahead, spd_ahead,
              road_ahead, last_valid_idx)
    """
    from scipy.interpolate import CubicSpline
    wp = np.asarray(plan_ego, np.float64).reshape(-1, 2)
    P = np.vstack([[0.0, 0.0], wp])
    t_src = np.r_[0.0, 0.5 * np.arange(1, len(P))]
    cs = CubicSpline(t_src, P, axis=0)
    tt = np.linspace(0.0, 4.0, N_REF)
    pos_ego = cs(tt)

    x, y, h = float(pose[0]), float(pose[1]), float(pose[2])
    c, s = math.cos(h), math.sin(h)
    pos_w = np.stack([x + pos_ego[:, 0] * c - pos_ego[:, 1] * s,
                      y + pos_ego[:, 0] * s + pos_ego[:, 1] * c], 1)

    d = np.diff(pos_ego, axis=0)
    heading_e = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    heading_e = np.r_[heading_e, heading_e[-1]] + h
    speed = np.r_[np.linalg.norm(d, axis=1) / DT, 0.0]
    speed = np.minimum(speed, v_max).astype(np.float32)

    road = road_dist_of_points(pos_w, seg_a, seg_b)

    # 曲率（三点法），向前预瞄取窗口均值
    kap = np.zeros(len(pos_w), np.float32)
    for i in range(1, len(pos_w) - 1):
        a = pos_w[i] - pos_w[i - 1]
        b = pos_w[i + 1] - pos_w[i]
        cross = a[0] * b[1] - a[1] * b[0]
        denom = max(np.linalg.norm(a) * np.linalg.norm(b)
                    * np.linalg.norm(pos_w[i + 1] - pos_w[i - 1]), 1e-6)
        kap[i] = float(abs(cross) / denom)
    kap[-1] = kap[-2] if len(kap) > 1 else 0.0

    # TODO: 以下三个"预瞄量"的精确口径尚未从 GPUDrive 版抄全
    # （`get_road_dist_batch` / `rebuild_paths_tensor` 里的定义待核对）
    w = 5                                   # 0.5 s 窗口
    curv_ahead = np.array([kap[min(i + w, len(kap) - 1)] for i in range(len(kap))],
                          np.float32)
    spd_ahead = np.array([speed[min(i + w, len(speed) - 1)] for i in range(len(speed))],
                         np.float32)
    road_ahead = np.array([road[min(i + w, len(road) - 1)] for i in range(len(road))],
                          np.float32)

    return dict(pos=pos_w.astype(np.float32),
                heading=heading_e.astype(np.float32),
                speed=speed, road_dist=road,
                curv_ahead=curv_ahead, spd_ahead=spd_ahead, road_ahead=road_ahead,
                last_valid_idx=N_REF - 1)


def others_from_world(ego_pose, actors_xyh_speed):
    """[(gx, gy, h, speed)] → 按距离升序，取 8 辆。"""
    x, y = float(ego_pose[0]), float(ego_pose[1])
    rows = []
    for gx, gy, gh, sp in actors_xyh_speed:
        rows.append((math.hypot(gx - x, gy - y), [gx, gy, gh, sp]))
    rows.sort(key=lambda r: r[0])
    return [r[1] for r in rows[:N_OTHER]]
