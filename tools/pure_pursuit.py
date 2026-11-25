"""纯追踪横向控制（几何解，天然稳定，速度自适应）

取代原来的 PID：原 PID 增益高 + 积分饱和 → steer 长期贴满舵（实测均值 -0.7）。
纯追踪：steer = atan(2·L·sin(α) / Ld)，Ld 随速度增长，无积分项。
"""
import math

import numpy as np

L_WHEELBASE = 2.9
MAX_STEER_RAD = math.radians(70.0)


class PurePursuit:
    def __init__(self):
        self.prev = 0.0
        self.integ = 0.0

    def step(self, route_pts, speed, pos, heading, dt=0.2):
        """route_pts: (N,2) 世界坐标（已含横向偏移）；返回 CARLA steer ∈ [-1,1]。"""
        if route_pts is None or len(route_pts) < 3:
            return 0.0
        P = np.asarray(route_pts, np.float64)
        v = max(float(speed), 0.0)

        # 前视距离随速度增长（低速 3 m，高速 12 m）
        k_look = float(__import__('os').environ.get('VEC_PP_KLOOK', '1.2'))
        lmin = float(__import__('os').environ.get('VEC_PP_LMIN', '3.0'))
        lmax = float(__import__('os').environ.get('VEC_PP_LMAX', '12.0'))
        Ld = float(np.clip(k_look * v + 3.0, lmin, lmax))

        # 沿路径找前视点
        d = np.linalg.norm(P - np.array([pos[0], pos[1]]), axis=1)
        j = int(np.argmin(d))
        seg = np.linalg.norm(np.diff(P[max(j - 1, 0):], axis=0), axis=1)
        s = np.r_[0.0, np.cumsum(seg)]
        target = P[min(j + int(np.searchsorted(s, Ld)) if len(s) > 1 else j, len(P) - 1)]

        dx, dy = target[0] - pos[0], target[1] - pos[1]
        # 目标点相对车头方向的角度（CARLA：y 右为正，heading 顺时针）
        alpha = math.atan2(dy, dx) - float(heading)
        alpha = math.atan2(math.sin(alpha), math.cos(alpha))

        # 纯追踪几何
        delta = math.atan2(2.0 * L_WHEELBASE * math.sin(alpha), max(Ld, 1.0))
        steer = delta / MAX_STEER_RAD

        # 小反馈：抑制横向偏差（不带积分，避免饱和）
        k_cte = float(__import__('os').environ.get('VEC_PP_KCTE', '0.15'))
        cte = -dx * math.sin(heading) + dy * math.cos(heading)
        steer += k_cte * cte / max(1.0 + v, 1.0)

        # 速率限制（每步最大变化）与低通，抑制抖动
        rate = float(__import__('os').environ.get('VEC_PP_RATE', '0.35'))
        lim = float(__import__('os').environ.get('VEC_PP_MAX', '0.75'))
        ds = float(np.clip(steer - self.prev, -rate, rate))
        steer = self.prev + ds
        self.prev = float(np.clip(steer, -lim, lim))
        return self.prev
