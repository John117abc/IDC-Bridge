import torch
import torch.nn as nn
import torch.nn.functional as F


# Actor网络输出层修改，避免饱和
class ContinuousActor(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # 加层归一化，稳定输出
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 2),
            nn.Tanh()
        )
        # 加速度通道给轻微正偏置: tanh(0.2)≈0.197 → a≈0.3 m/s² 怠速蠕行
        # 转向通道保持零偏置
        nn.init.constant_(self.net[6].bias, 0.0)
        self.net[6].bias.data[1] = 0.2

    def forward(self, x):
        return self.net(x)



class ContinuousCritic(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, output_dim: int = 1):
        super().__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.elu(self.l1(x))
        x = F.elu(self.l2(x))
        x = self.l3(x)
        return x