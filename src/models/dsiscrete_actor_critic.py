import torch
import torch.nn as nn
import torch.nn.functional as F


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


class DiscreteActor(nn.Module):
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


class DiscreteCritic(nn.Module):
    """Actor-Critic 中的 Critic 网络，评估状态价值 V(s)"""
    def __init__(self, obs_dim, hidden_dim=256):
        super().__init__()
        self.backbone = DrivingBackbone(obs_dim, hidden_dim)   # 独立的 backbone
        self.value_head = nn.Linear(hidden_dim, 1)             # 输出 V(s)

    def forward(self, obs):
        features = self.backbone(obs)     # [B, hidden_dim]
        value = self.value_head(features) # [B, 1]
        return value