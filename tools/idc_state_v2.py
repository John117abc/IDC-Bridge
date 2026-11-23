"""IDC v2 状态 = v1 核心（66 维，保留绝对坐标）+ 追加特征（19 维）= 85 维

为什么保留 v1 核心布局：
  IDC 的状态**故意存绝对坐标**（x, y, heading 在前 6 维），帧不变性由 `_encode`
  在进入网络前完成。训练时的动力学推演（预测下一状态）和奖励（查参考点）
  都依赖绝对坐标，所以不能去掉。

v2 追加的 19 维（都放在“veh 之前”，保证 `states[..., -1]` 仍是车长）：

    65:67   ego extras        [a_lat/5, steer_prev]
    67:83   others extras     8 辆 × [ttc/10, in_corridor]
    83:84   replan phase      (step % 5) / 5

完整布局（85）：
    0:6    ego core        [x, y, v, 0, h, 0]               绝对坐标
    6:38   others core     8 × [gx, gy, gh, v]              绝对坐标
    38:46  validity        8
    46:64  ref_error       18  (v1 语义：dp,dphi,dv + 3×[lat,dphi,road,spd] + 3)
    64:65  temporal        t（参考索引）
    65:67  ego extras
    67:83  others extras
    83:84  replan phase
    84:85  veh_len
"""
import math

import numpy as np

V1_DIM = 66
EXTRA_EGO = 2
EXTRA_OTHER = 2                 # ttc, in_corridor
N_OTHER = 8
EXTRA_DIM = EXTRA_EGO + N_OTHER * EXTRA_OTHER            # 18
V2_DIM = V1_DIM + EXTRA_DIM     # 84
CORRIDOR_HALF_W = 2.0

SCALE_A_LAT = 5.0
SCALE_TTC = 10.0


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def neighbor_extras(pose, neighbors, v_ego=0.0):
    """8 辆周车的 [ttc/10, in_corridor]（ego 系判断，坐标系无关）。"""
    x, y, h = float(pose[0]), float(pose[1]), float(pose[2])
    c, s = math.cos(-h), math.sin(-h)
    out = np.zeros((N_OTHER, EXTRA_OTHER), np.float32)
    for k, nb in enumerate(neighbors[:N_OTHER]):
        dx, dy = float(nb['x']) - x, float(nb['y']) - y
        rx = dx * c - dy * s
        ry = dx * s + dy * c
        d = math.hypot(dx, dy)
        if d < 1e-6:
            continue
        # 接近速度：对方朝我方向的分量 + 自车速度
        vx = float(nb.get('v', 0.0)) * math.cos(float(nb.get('h', 0.0)))
        vy = float(nb.get('v', 0.0)) * math.sin(float(nb.get('h', 0.0)))
        closing = -(vx * dx + vy * dy) / max(d, 1e-3) + float(v_ego)
        ttc = min(d / max(closing, 0.5), SCALE_TTC)
        out[k, 0] = ttc / SCALE_TTC
        out[k, 1] = 1.0 if (abs(ry) <= CORRIDOR_HALF_W and rx > 0) else 0.0
    return out.reshape(-1)


def build_v2(core66, pose, neighbors, a_lat=0.0, steer_prev=0.0,
             step=0, replan_every=5, v_ego=0.0):
    """core66 → 85 维（把追加特征插在 veh 之前）。"""
    core = np.asarray(core66, np.float32).reshape(-1)
    assert core.shape[0] == V1_DIM, core.shape
    ego_ex = np.array([a_lat / SCALE_A_LAT, float(steer_prev)], np.float32)
    oth_ex = neighbor_extras(pose, neighbors, v_ego=v_ego)
    out = np.concatenate([core[:V1_DIM - 1], ego_ex, oth_ex,
                          core[V1_DIM - 1:]]).astype(np.float32)
    assert out.shape[0] == V2_DIM, out.shape
    return out
