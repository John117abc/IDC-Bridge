import numpy as np
import torch
from typing import Tuple

# ==============================================
# 离散动作映射工具
# ==============================================
class DiscreteActionMapper:
    """
    将 GPUDrive 的离散动作索引 (0 ~ 90) 映射到连续物理量 (accel, steer)，
    并支持反向查表（用于前向预测时从连续量找到最近的离散动作）。
    """
    def __init__(self, steer_bins=13, accel_bins=7,
                 steer_range=(-0.4, 0.4), accel_range=(-3.0, 1.5)):
        self.steer_bins = steer_bins
        self.accel_bins = accel_bins
        self.steer_edges = torch.linspace(steer_range[0], steer_range[1], steer_bins)
        self.accel_edges = torch.linspace(accel_range[0], accel_range[1], accel_bins)

    def index_to_action(self, steer_idx: int, accel_idx: int) -> np.ndarray:
        """将离散索引转为物理动作 [accel, steer]"""
        accel = self.accel_edges[accel_idx].item()
        steer = self.steer_edges[steer_idx].item()
        return np.array([accel, steer], dtype=np.float32)

    def action_to_index(self, accel: float, steer: float) -> Tuple[int, int]:
        """将物理动作映射回最近的离散索引"""
        steer_idx = torch.argmin(torch.abs(self.steer_edges - steer)).item()
        accel_idx = torch.argmin(torch.abs(self.accel_edges - accel)).item()
        return steer_idx, accel_idx

    def full_action_idx_to_single(self, idx: int) -> Tuple[int, int]:
        """
        GPUDrive 通常把 (steer, accel) 展平为一维索引：
        总动作数 = steer_bins * accel_bins
        返回 (steer_idx, accel_idx)
        """
        steer_idx = idx // self.accel_bins
        accel_idx = idx % self.accel_bins
        return steer_idx, accel_idx