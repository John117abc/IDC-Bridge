import torch
import torch.nn as nn

class DrivingBackbone(nn.Module):
    """共享的主干网络：提取场景特征"""
    def __init__(self, obs_dim, hidden_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, obs):
        return self.encoder(obs)  # 输出一个抽象的状态表征


class DiscretePolicy(nn.Module):
    """GPUDrive 离散动作策略"""
    def __init__(self, obs_dim, hidden_dim=256, steer_bins=13, accel_bins=7):
        super().__init__()
        self.backbone = DrivingBackbone(obs_dim, hidden_dim)
        self.steer_head = nn.Linear(hidden_dim, steer_bins)
        self.accel_head = nn.Linear(hidden_dim, accel_bins)

    def forward(self, obs):
        features = self.backbone(obs)
        steer_logits = self.steer_head(features)
        accel_logits = self.accel_head(features)
        return steer_logits, accel_logits


class ContinuousPolicy(nn.Module):
    """CARLA 连续动作策略"""
    def __init__(self, obs_dim, hidden_dim=256):
        super().__init__()
        self.backbone = DrivingBackbone(obs_dim, hidden_dim)
        self.steer_mean = nn.Linear(hidden_dim, 1)
        self.steer_std = nn.Linear(hidden_dim, 1)
        self.accel_mean = nn.Linear(hidden_dim, 1)
        self.accel_std = nn.Linear(hidden_dim, 1)

    def forward(self, obs):
        features = self.backbone(obs)
        steer = torch.tanh(self.steer_mean(features))      # [-1, 1]
        accel = torch.tanh(self.accel_mean(features))       # [-1, 1]
        return steer, accel