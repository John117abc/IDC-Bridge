import torch
import torch.nn as nn


class KinematicBicycleModel(nn.Module):
    """运动学自行车模型（以重心为参考点），与 GPUDrive `forwardKinematics` 同构。

    GPUDrive 的更新式（`src/dynamics.hpp`）：
        dt      = 0.1
        tanD    = tan(steering)
        beta    = atan(0.5 * tanD)                  # 0.5 = lr/L，重心在车长中点
        w       = v * cos(beta) * tanD / size.length
        yaw    += w * dt
        speed  += a * dt

    关键点：**轴距用的是 size.length（整车长度），而且逐车不同**。
    实测该值等于观测张量 `absolute_self_observation_tensor()[..., 10]`
    （4.80~5.29 m 都出现过），所以这里接受按样本传入的 wheelbase。
    """

    def __init__(self, dt: float = 0.1, L: float = 5.0,
                 lr_ratio: float = 0.5, v_max: float = 30.0):
        super().__init__()
        self.dt = float(dt)
        self.L = float(L)          # 缺省轴距（拿不到逐车值时兜底）
        self.lr_ratio = float(lr_ratio)
        self.v_max = float(v_max)

    def forward(self, state: torch.Tensor, action: torch.Tensor,
                wheelbase: torch.Tensor = None) -> torch.Tensor:
        """
        state: [..., 6] = [x, y, v_lon, v_lat, phi, omega]
        action: [..., 2] = [a(m/s²), delta(rad)]
        wheelbase: None | [B] | [B,1]，逐样本轴距；None 时用 self.L
        返回: [..., 4] = [x, y, phi, v]
        """
        vx, vy = state[..., 2], state[..., 3]
        x, y, theta = state[..., 0], state[..., 1], state[..., 4]
        v = torch.sqrt(vx ** 2 + vy ** 2 + 1e-6)
        a, delta = action[..., 0], action[..., 1]

        if wheelbase is None:
            L = torch.full_like(v, self.L)
        else:
            L = wheelbase.reshape(-1).to(v.device).float()
            L = L.expand_as(v) if L.numel() == 1 else L
        L = torch.clamp(L, min=0.5)

        # 1. 半隐式平均速度
        v_bar = torch.clamp(v + 0.5 * a * self.dt, -self.v_max, self.v_max)

        # 2. 滑移角（与 L 无关：beta = atan((lr/L)·tanδ) = atan(lr_ratio·tanδ)）
        tan_delta = torch.tan(delta)
        beta = torch.atan(self.lr_ratio * tan_delta)

        # 3. 位置导数与横摆角速度
        x_dot = v_bar * torch.cos(theta + beta)
        y_dot = v_bar * torch.sin(theta + beta)
        theta_dot = v_bar * torch.cos(beta) * tan_delta / L

        # 4. 欧拉积分
        x_next = x + x_dot * self.dt
        y_next = y + y_dot * self.dt
        theta_next = theta + theta_dot * self.dt
        v_next = torch.clamp(v + a * self.dt, -self.v_max, self.v_max)
        return torch.stack([x_next, y_next, theta_next, v_next], dim=-1)
