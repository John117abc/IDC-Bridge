import numpy as np

import math
from typing import Tuple

def batch_world_to_ego(path_locations, ego_transform):
    xy_world = np.array([[p.x, p.y] for p in path_locations], dtype=np.float32)
    ego_x = ego_transform.location.x
    ego_y = ego_transform.location.y
    yaw = np.radians(ego_transform.rotation.yaw)
    c, s = np.cos(yaw), np.sin(yaw)

    dx = xy_world[:, 0] - ego_x
    dy = xy_world[:, 1] - ego_y

    # 修复1：使用正确的旋转矩阵（注意第二行符号）
    x_ego = dx * c + dy * s
    y_ego = dx * (-s) + dy * c  # 修复符号问题

    # 修复2：确保横向误差定义正确
    # 在IDC中：y_ego > 0 表示参考路径在车辆左侧（需要右转）
    #          y_ego < 0 表示参考路径在车辆右侧（需要左转）
    return np.stack([x_ego, y_ego], axis=1).tolist()