"""PID 执行层（对齐 PlanT 的 LateralPIDController / LongitudinalPIDController）

设计：
  · 网络（IDC）只出**意图**：曲率意图 δ、纵向加速度意图 a
  · 本模块把意图变成 steer / throttle / brake：
      - 横向：把 route 按 δ 横移 off = d_preview · tan(δ)，再用 PlanT 调好的
              PD+前视 跟踪这条横移后的路径
      - 纵向：把 v_ref + a 的修正作为目标速度，用 PlanT 的 PID（km/h 域，
              braking_ratio 规则）出油门/刹车
  · 参数取自 carla_garage/config.py（贝叶斯优化过的值）
"""
import math
import os

import numpy as np

# ---- PlanT 调好的参数 ----
LAT_KP = 3.118357247806046
LAT_KD = 1.3782508892109167
LAT_KI = 0.6406067986034124
LAT_LOOK_MIN = 2.4          # m
LAT_LOOK_MAX = 10.5         # m
LAT_LOOK_SCALE = 0.9755     # m per m/s
LAT_LOOK_OFFSET = 1.9153

LON_KP = 1.0016429066823955
LON_KD = 1.5761818624794222
LON_KI = 0.2941563856687906
LON_BRAKE_RATIO = 1.0324622059220139
LON_MIN_TARGET = 0.278      # m/s
LON_DT = 0.05               # 20 Hz


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class LateralPID:
    def __init__(self):
        self.err_hist = []
        self.prev_err = None
        self.integ = 0.0
        # 误差已归一化到弧度，增益按“0.1 rad → steer≈0.2”标定
        self.kp = float(os.environ.get('VEC_LAT_KP', '1.6'))
        self.kd = float(os.environ.get('VEC_LAT_KD', '0.25'))
        self.ki = float(os.environ.get('VEC_LAT_KI', '0.05'))
        self.max_steer = float(os.environ.get('VEC_LAT_MAX', '0.6'))

    def reset(self):
        self.err_hist = []
        self.prev_err = None
        self.integ = 0.0

    def step(self, route_pts, speed, pos, heading, dt=LON_DT):
        """route_pts: (N,2) 世界坐标；返回 steer ∈ [-1,1]。"""
        if route_pts is None or len(route_pts) < 3:
            return 0.0
        look = LAT_LOOK_SCALE * float(speed) + LAT_LOOK_OFFSET
        look = float(np.clip(look, LAT_LOOK_MIN, LAT_LOOK_MAX))
        # 取前进方向上的第 look 个点（route 点间距约 2 m）
        seg = np.linalg.norm(np.diff(route_pts, axis=0), axis=1).mean()
        idx = int(np.clip(round(look / max(seg, 1e-3)), 1,
                          len(route_pts) - 1))
        tgt = route_pts[idx]
        dx, dy = tgt[0] - pos[0], tgt[1] - pos[1]
        alpha = wrap(math.atan2(dy, dx) - heading)     # 期望航向误差

        # 横向偏差（以航向为基准的侧向距离）
        cte = -dx * math.sin(heading) + dy * math.cos(heading)
        # 误差量纲统一到“弧度”：航向误差为主，横向偏差按前视距离折算
        err = alpha + math.atan2(cte, max(look, 1.0))

        d_err = 0.0 if self.prev_err is None else (err - self.prev_err) / max(dt, 1e-3)
        self.prev_err = err
        self.integ = float(np.clip(self.integ + err * dt, -0.5, 0.5))
        steer = self.kp * err + self.kd * d_err + self.ki * self.integ
        return float(np.clip(steer, -self.max_steer, self.max_steer))


class LongitudinalPID:
    def __init__(self):
        self.err_hist = []
        self.prev_err = None

    def reset(self):
        self.err_hist = []
        self.prev_err = None

    def step(self, target_speed, speed, hazard=False, dt=LON_DT):
        """返回 (throttle, brake) ∈ [0,1]。"""
        if hazard or target_speed < 1e-5:
            return 0.0, 1.0
        tgt = max(LON_MIN_TARGET, float(target_speed))
        v_now, v_tgt = 3.6 * float(speed), 3.6 * tgt      # km/h
        if v_tgt > 1e-3 and v_now / v_tgt > LON_BRAKE_RATIO:
            return 0.0, 1.0
        err = v_tgt - v_now
        d_err = 0.0 if self.prev_err is None else (err - self.prev_err) / max(dt, 1e-3)
        self.prev_err = err
        u = LON_KP * err + LON_KD * d_err + LON_KI * np.clip(err * dt, -5, 5)
        # 油门标定：err=10 km/h 时约 0.5 油门（原来除以 100 太弱，车只能爬行）
        gain = float(os.environ.get('VEC_LON_SCALE', '20.0'))
        throttle = float(np.clip(u / gain, 0.0, 1.0))
        brake = float(np.clip(-u / gain, 0.0, 1.0)) if u < -2.0 else 0.0
        return throttle, brake


def shift_path(pts, off):
    """把路径沿法向整体平移 off 米（正=左侧）。"""
    if pts is None or len(pts) < 2:
        return pts
    P = np.asarray(pts, np.float64)
    d = np.gradient(P, axis=0)
    n = np.stack([-d[:, 1], d[:, 0]], 1)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-6)
    return P + n * float(off)
