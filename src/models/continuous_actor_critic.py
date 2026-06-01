import torch
import torch.nn as nn


class ContinuousActor(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        h = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, h),
            nn.LayerNorm(h),
            nn.ELU(),
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.ELU(),
            nn.Linear(h, h // 2),
            nn.LayerNorm(h // 2),
            nn.ELU(),
            nn.Linear(h // 2, 2),
        )
        nn.init.constant_(self.net[-1].bias[0], 0.0)
        nn.init.constant_(self.net[-1].bias[1], 1.2)

    def forward(self, x):
        return self.net(x)


class ContinuousCritic(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, output_dim: int = 1):
        super().__init__()
        h = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, h),
            nn.LayerNorm(h),
            nn.ELU(),
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.ELU(),
            nn.Linear(h, h // 2),
            nn.LayerNorm(h // 2),
            nn.ELU(),
            nn.Linear(h // 2, output_dim),
        )

    def forward(self, x):
        return self.net(x)
