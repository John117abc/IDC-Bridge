import torch

import torch
import torch.nn as nn

class KinematicBicycleModel(nn.Module):
    """
    运动学自行车模型 (Kinematic Bicycle Model)，以重心为参考点。
    参考：Nocturne 论文 Appendix D

    状态量 (state): [x, y, theta, v] 
        - x, y: 重心在全局坐标系中的位置 (m)
        - theta: 航向角 (rad)，与全局 x 轴夹角
        - v: 速度 (m/s)，沿车身纵轴方向，向前为正
    控制量 (action): [a, delta]
        - a: 加速度 (m/s^2)
        - delta: 前轮转向角 (rad)，左正右负

    更新公式 (dt 固定，半隐式欧拉，带速度限幅)：
        v_dot = a
        v_bar = clip(v_t + 0.5 * v_dot * dt, -v_max, v_max)
        beta = arctan(lr * tan(delta) / L)
        x_dot = v_bar * cos(theta + beta)
        y_dot = v_bar * sin(theta + beta)
        theta_dot = v_bar * cos(beta) * tan(delta) / L
        x_{t+1} = x_t + x_dot * dt
        y_{t+1} = y_t + y_dot * dt
        theta_{t+1} = theta_t + theta_dot * dt
        v_{t+1} = clip(v_t + v_dot * dt, -v_max, v_max)
    """

    def __init__(
        self,
        dt: float = 0.1,
        L: float = 4.0,
        lr_ratio: float = 0.5,
        v_max: float = 30.0,
    ):
        """
        Args:
            dt: 离散时间步长 (s)，默认为 0.1 (对应 Nocturne 的 10 Hz)
            L: 轴距 (m)，即前轴到后轴的距离。若不确知，可设为典型值（如 4.0）
            lr_ratio: 重心到后轴的距离占轴距的比例，论文固定为 0.5
            v_max: 最大允许速度 (m/s)，用于限幅，防止数值发散
        """
        super().__init__()
        self.dt = dt
        self.L = L
        self.lr = lr_ratio * L   # 重心到后轴距离
        self.v_max = v_max

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播：计算下一个状态。

        Args:
            state: 张量 shape (..., 4)，最后维为 [x, y, theta, v]
            action: 张量 shape (..., 2)，最后维为 [a, delta]

        Returns:
            next_state: 张量 shape (..., 4)，与 state 同型
        """
        vx, vy = state[..., 2], state[..., 3]
        x, y, theta, v = state[..., 0], state[..., 1], state[..., 4], torch.sqrt(vx**2 + vy**2)
        a, delta = action[..., 0], action[..., 1]

        dt = self.dt
        L = self.L
        lr = self.lr
        v_max = self.v_max

        # 1. 加速度积分的平均速度 (半隐式)
        v_dot = a
        v_bar = v + 0.5 * v_dot * dt
        v_bar = torch.clamp(v_bar, -v_max, v_max)

        # 2. 滑移角 beta
        tan_delta = torch.tan(delta)
        beta = torch.atan(lr * tan_delta / L)   # 论文中 lr/L = 0.5

        # 3. 位置导数
        cos_theta_beta = torch.cos(theta + beta)
        sin_theta_beta = torch.sin(theta + beta)
        x_dot = v_bar * cos_theta_beta
        y_dot = v_bar * sin_theta_beta

        # 4. 航向角导数
        theta_dot = v_bar * torch.cos(beta) * tan_delta / L

        # 5. 欧拉积分更新
        x_next = x + x_dot * dt
        y_next = y + y_dot * dt
        theta_next = theta + theta_dot * dt
        v_next = v + v_dot * dt
        v_next = torch.clamp(v_next, -v_max, v_max)

        # 合并输出
        next_state = torch.stack([x_next, y_next, theta_next, v_next], dim=-1)
        return next_state