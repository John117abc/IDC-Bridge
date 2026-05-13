import numpy as np
import math
from typing import List, Dict, Tuple, Optional
import torch

from utils import get_logger
logger = get_logger('per_buffer')


class GPUDriveObservationBuilder:
    """
    完全基于 gpudrive 环境自带的观测接口构建 IDC 所需向量。
    依赖：
        - env.get_obs() 返回的字典（包含 road_map_obs, partner_obs 等）
        - env.sim.absolute_obs() （仅用于自车绝对位置/速度，因 ego_state 不含这些）
    """

    def __init__(self, env, scene_json_list: List[Dict]):
        """
        :param env: GPUDriveTorchEnv 实例（或任何提供 get_obs() 和 sim 的对象）
        :param scene_json_list: 保留以备目标点回退，不再用于道路点提取
        """
        self.env = env
        self.sim = env.sim
        self.num_worlds = env.num_worlds

        # 步数计数器
        self.step_counter = {w: 0 for w in range(self.num_worlds)}

        # 缓存目标点
        self.goal_positions = []
        for scene in scene_json_list:
            goals = {}
            for obj in scene.get('objects', []):
                aid = obj.get('id', -1)
                if aid != -1 and 'goalPosition' in obj:
                    gp = obj['goalPosition']
                    goals[aid] = (gp['x'], gp['y'])
            self.goal_positions.append(goals)

        # 专家轨迹（如果存在）
        self.expert_traj = None
        self.EXPERT_TRAJ_LEN = None
        if hasattr(self.sim, 'expert_trajectory_tensor'):
            try:
                self.expert_traj = self.sim.expert_trajectory_tensor().to_torch()
                self.EXPERT_TRAJ_LEN = self.expert_traj.shape[2] // 16
                logger.info(f"Expert trajectory available, shape: {self.expert_traj.shape}")
            except Exception as e:
                logger.warning(f"Unable to get expert_trajectory_tensor: {e}")

    def reset_world_step(self, world_idx: int, step: int = 0):
        self.step_counter[world_idx] = step

    def increment_step(self, world_idx: int):
        self.step_counter[world_idx] += 1

    # ====================== 主入口 ======================
    def get_idc_observation(self, world_idx: int, agent_idx: int,
                            perceived_distance: float = 30.0,
                            num_other_vehicles: int = 8,
                            num_road_points: int = 20) -> Tuple[np.ndarray, ...]:
        """
        返回:
            network_state: (6 + num_other_vehicles*6 + num_road_points*6*2 + 3)
            s_road: (num_road_points*2*6,) 左右道路边缘展平向量
            s_ref_raw: (6,)
            s_ref_error: (3,)
            s_other: (num_other_vehicles, 6)
        """
        # 获取最新观测
        obs = self.env.get_obs()  # dict with 'road_map_obs', 'partner_obs', 'ego_state'

        s_ego = self._get_ego_state(world_idx, agent_idx)                # (6,)
        s_other = self._get_other_vehicles(world_idx, agent_idx, num_other_vehicles)
        s_road = self._get_road_edges(world_idx, agent_idx,              # (num_points*2*6,)
                                      obs, s_ego, num_road_points, perceived_distance)
        s_ref = self._get_ref_state(world_idx, agent_idx, s_ego)     # (6,)
        s_ref_error = self._calc_ref_error(s_ego, s_ref)             # (3,)

        network_state = np.concatenate([
            s_ego,
            s_other.flatten().astype(np.float32),
            s_road.astype(np.float32),
            s_ref_error.astype(np.float32)
        ])
        return network_state, s_road, s_ref, s_ref_error, s_other

    # --------------------------------------------------------
    #  自车状态 [x, y, vx, vy, heading, omega=0]
    #  注意：ego_state 不提供位置和分速度，只能用 absolute_obs
    # --------------------------------------------------------
    def _get_ego_state(self, world_idx: int, agent_idx: int) -> np.ndarray:
        # 绝对观测：位置 + 航向
        abs_tensor = self.sim.absolute_self_observation_tensor().to_torch()
        state_abs = abs_tensor[world_idx, agent_idx].cpu().numpy()
        x = state_abs[0]
        y = state_abs[1]
        heading = state_abs[7]
        
        # 相对观测：速度（标量）
        rel_tensor = self.sim.self_observation_tensor().to_torch()
        speed = rel_tensor[world_idx, agent_idx, 0].item()
        
        vx = speed * np.cos(heading)
        vy = speed * np.sin(heading)
        omega = 0.0  # 角速度不直接提供
        
        return np.array([x, y, vx, vy, heading, omega], dtype=np.float32)



    def _get_other_vehicles(self, 
                            world_idx: int, 
                            ego_agent_idx: int, 
                            max_partners: int = None
                            ) -> np.ndarray:
        """
        获取周车状态（全局绝对坐标），按距离自车由近到远排序。
        
        Args:
            world_idx: 世界索引
            ego_agent_idx: 自车在场景中的索引（通常是0或可控智能体索引）
            max_partners: 最多返回多少个周车，默认None表示返回全部
        
        Returns:
            np.ndarray: 形状 (num_partners, 6)，每行 [x, y, vx, vy, heading, omega]
                        按距离自车升序排列
        """
        # 1. 获取自车状态（用于坐标转换和距离计算）
        ego_state = self._get_ego_state(world_idx, ego_agent_idx)  # [x, y, vx, vy, heading, omega]
        ego_x, ego_y = ego_state[0], ego_state[1]
        ego_heading = ego_state[4]
        
        # 2. 获取周车观测张量 (num_worlds, max_agents, max_agents-1, 9)
        partner_tensor = self.sim.partner_observation_tensor().to_torch()
        # 取出当前世界和当前自车对应的周车数据
        # 形状: (max_agents-1, 9)
        partners = partner_tensor[world_idx, ego_agent_idx].cpu().numpy()
        
        num_partners = partners.shape[0]  # = max_agents - 1
        if num_partners == 0:
            return np.empty((0, 6), dtype=np.float32)
        
        # 3. 解析每个周车
        partner_states = []
        for i in range(num_partners):
            p = partners[i]
            # 根据 PartnerObs 的解析顺序（observation.py）:
            # 0:speed, 1:rel_pos_x, 2:rel_pos_y, 3:orientation,
            # 4:vehicle_length, 5:vehicle_width, 6:vehicle_height,
            # 7:agent_type, 8:ids
            speed = p[0]
            rel_x = p[1]
            rel_y = p[2]
            rel_heading = p[3]   # 相对朝向角（弧度）
            
            # 绝对朝向角
            abs_heading = ego_heading + rel_heading
            
            # 将相对坐标转换为全局绝对坐标
            # 旋转矩阵: 自车坐标系 -> 全局坐标系
            cos_h = np.cos(ego_heading)
            sin_h = np.sin(ego_heading)
            global_x = ego_x + rel_x * cos_h - rel_y * sin_h
            global_y = ego_y + rel_x * sin_h + rel_y * cos_h
            
            # 计算速度分量
            vx = speed * np.cos(abs_heading)
            vy = speed * np.sin(abs_heading)
            
            # 距离（用于排序）
            distance = np.hypot(global_x - ego_x, global_y - ego_y)
            
            partner_states.append((distance, [global_x, global_y, vx, vy, abs_heading, 0.0]))
        
        # 4. 按距离排序
        partner_states.sort(key=lambda x: x[0])
        
        # 5. 截取前 max_partners 个
        if max_partners is not None:
            partner_states = partner_states[:max_partners]
        
        # 6. 转换为 numpy 数组
        result = np.array([state for _, state in partner_states], dtype=np.float32)
        return result

    def _get_left_right_road_boundary(self, world_idx: int, agent_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        根据自车位置和朝向，找到垂直于行驶方向左右两侧最近的 RoadLine 点。
        返回格式: (left_boundary_point, right_boundary_point)
        每个点的形式: [x, y, 0, 0, 0, 0] (其中 x, y 是道路点的绝对坐标)
        如果某一侧找不到任何点，对应返回值变为 [inf, inf, 0, 0, 0, 0]。
        """
        # 获取自车状态 (绝对坐标 [x, y, heading])
        self_state = self._get_ego_state(world_idx, agent_idx)
        ego_x, ego_y, ego_heading = self_state[0], self_state[1], self_state[4]  # ego_heading (yaw)

        # 获取道路观测数据
        # road_map_obs_tensor 形状: (num_worlds, max_agents, top_k, 13)
        # 假设此函数所在类能获取到 road_tensor
        road_tensor = self.sim.road_observation_tensor().to_torch()  # 如果环境没有显式提供，可能需要从其他方式获取
        road_points = road_tensor[world_idx, agent_idx]  # 取出当前 agent 观测到的道路点 (top_k, 13)
        road_points_np = road_points.cpu().numpy()

        # 定义常量：点类型在索引 12 处 (0: None, 1: RoadLine, ...)
        TYPE_IDX = 12
        ROADLINE_TYPE = 1

        # 筛选出所有类型为 RoadLine 的点
        line_points = road_points_np[road_points_np[:, TYPE_IDX] == ROADLINE_TYPE]
        if len(line_points) == 0:
            # 没有找到任何车道线点
            inf_point = np.array([float('inf'), float('inf'), 0., 0., 0., 0.], dtype=np.float32)
            return (inf_point.copy(), inf_point.copy())

        # 提取坐标 (绝对坐标已提供)
        xs = line_points[:, 0]
        ys = line_points[:, 1]

        # 计算每个点到自车位置的向量差 (绝对坐标系)
        dx = xs - ego_x
        dy = ys - ego_y

        # 将车辆朝向作为前进方向，定义左侧和右侧的方向向量
        # 前进方向单位向量 (vehicle forward)
        forward_x = np.cos(ego_heading)
        forward_y = np.sin(ego_heading)
        # 左侧方向向量: 逆时针旋转 90 度: (-forward_y, forward_x)
        left_dir_x = -forward_y
        left_dir_y = forward_x
        # 右侧方向向量: 顺时针旋转 90 度: (forward_y, -forward_x)
        right_dir_x = forward_y
        right_dir_y = -forward_x

        # 计算每个点到自车位置的向量在左右方向上的投影距离 (带符号)
        # 投影到左侧方向的正值表示点位于车辆左侧
        proj_left = dx * left_dir_x + dy * left_dir_y
        proj_right = dx * right_dir_x + dy * right_dir_y

        # 只考虑左右方向投影为正值的点 (即位于该侧)
        left_mask = proj_left > 0
        right_mask = proj_right > 0

        # 如果没有点位于某一侧，返回无穷大点
        inf_point = np.array([float('inf'), float('inf'), 0., 0., 0., 0.], dtype=np.float32)

        if not np.any(left_mask):
            left_closest = inf_point.copy()
        else:
            # 取左侧投影距离最小的点 (即离车辆横向最近)
            left_idx = np.argmin(proj_left[left_mask])
            left_global_idx = np.where(left_mask)[0][left_idx]
            left_x = line_points[left_global_idx, 0]
            left_y = line_points[left_global_idx, 1]
            left_closest = np.array([left_x, left_y, 0., 0., 0., 0.], dtype=np.float32)

        if not np.any(right_mask):
            right_closest = inf_point.copy()
        else:
            # 取右侧投影距离最小的点
            right_idx = np.argmin(proj_right[right_mask])
            right_global_idx = np.where(right_mask)[0][right_idx]
            right_x = line_points[right_global_idx, 0]
            right_y = line_points[right_global_idx, 1]
            right_closest = np.array([right_x, right_y, 0., 0., 0., 0.], dtype=np.float32)

        return left_closest, right_closest

    # --------------------------------------------------------
    #  参考路径
    # --------------------------------------------------------
    def _get_ref_state(self, world_idx: int, agent_idx: int,
                       ego_state: np.ndarray,
                       obs_dict: Dict,
                       default_speed: float = 10.0) -> np.ndarray:
        """
        专家参考状态：[x_ref, y_ref, vx_ref, 0, heading_ref, 0]
        1. 优先使用专家轨迹的未来点；
        2. 否则从 road_map_obs 中提取车道中心线 (RoadLane) 计算参考方向及参考点；
        3. 若仍无结果，回退到目标点直线方向。
        """
        # ---------- 专家轨迹 ----------
        if self.expert_traj is not None:
            step = self.step_counter[world_idx]
            if step < self.EXPERT_TRAJ_LEN:
                traj = self.expert_traj[world_idx]  # [num_agents, T*16]
                T = self.EXPERT_TRAJ_LEN
                pos = traj[:, :2 * T].reshape(-1, T, 2)
                vel = traj[:, 2 * T:4 * T].reshape(-1, T, 2)
                heading = traj[:, 4 * T:5 * T].reshape(-1, T)
                ref_x = pos[agent_idx, step, 0].item()
                ref_y = pos[agent_idx, step, 1].item()
                vx = vel[agent_idx, step, 0].item()
                ref_heading = heading[agent_idx, step].item()
                return np.array([ref_x, ref_y, vx, 0.0, ref_heading, 0.0], dtype=np.float32)

        # ---------- 车道中心线 (RoadLane) ----------
        road_map = obs_dict['road_map_obs']  # (worlds, max_agents, top_k, 13)
        points = road_map[world_idx, agent_idx]  # (top_k, 13) 相对坐标
        if points.shape[0] > 0:
            TYPE_IDX = 6  # 点类型字段索引，请根据实际调整
            lane_mask = points[:, TYPE_IDX].cpu().numpy() == 3  # RoadLane = 3
            if np.any(lane_mask):
                lane_pts = points[lane_mask]  # (N, 13)
                rel_x = lane_pts[:, 0].cpu().numpy()
                rel_y = lane_pts[:, 1].cpu().numpy()
                # 转换到车辆坐标系计算纵向距离
                cos_h = math.cos(ego_state[4])
                sin_h = math.sin(ego_state[4])
                proj_long = rel_x * cos_h + rel_y * sin_h
                # 只取前方车道点
                front_mask = proj_long > 0
                if np.any(front_mask):
                    # 取纵向距离最小的点作为参考点
                    idx = np.argmin(proj_long[front_mask])
                    # 因为 front_mask 是布尔数组，需要映射回原索引
                    front_indices = np.where(front_mask)[0]
                    best_idx = front_indices[idx]
                    rx = rel_x[best_idx]
                    ry = rel_y[best_idx]
                    # 计算绝对参考位置
                    ref_x = ego_state[0] + rx
                    ref_y = ego_state[1] + ry
                    # 估算车道方向：取该点及其前方第二个点（如果有）
                    if np.sum(front_mask) >= 2:
                        sorted_front = np.argsort(proj_long[front_mask])
                        second_idx = front_indices[sorted_front[1]]
                        rx2 = rel_x[second_idx]
                        ry2 = rel_y[second_idx]
                        dx = rx2 - rx
                        dy = ry2 - ry
                    else:
                        # 只有一个点，用自车朝向
                        dx = math.cos(ego_state[4])
                        dy = math.sin(ego_state[4])
                    ref_heading = math.atan2(dy, dx)
                    return np.array([ref_x, ref_y, default_speed, 0.0, ref_heading, 0.0], dtype=np.float32)

        # ---------- 终极回退：目标点方向 ----------
        goals = self.goal_positions[world_idx]
        gx, gy = goals.get(agent_idx, (ego_state[0], ego_state[1]))
        dx = gx - ego_state[0]
        dy = gy - ego_state[1]
        dist = math.hypot(dx, dy)
        if dist > 1e-3:
            ref_heading = math.atan2(dy, dx)
            ref_x = ego_state[0] + 3.0 * math.cos(ref_heading)
            ref_y = ego_state[1] + 3.0 * math.sin(ref_heading)
        else:
            ref_heading = ego_state[4]
            ref_x, ref_y = gx, gy
        return np.array([ref_x, ref_y, default_speed, 0.0, ref_heading, 0.0], dtype=np.float32)

    @staticmethod
    def _calc_ref_error(ego_state: np.ndarray, ref_state: np.ndarray) -> np.ndarray:
        dx = ref_state[0] - ego_state[0]
        dy = ref_state[1] - ego_state[1]
        delta_p = math.hypot(dx, dy)
        cross = dy * math.cos(ref_state[4]) - dx * math.sin(ref_state[4])
        delta_p *= math.copysign(1.0, cross)

        delta_phi = ego_state[4] - ref_state[4]
        delta_phi = math.atan2(math.sin(delta_phi), math.cos(delta_phi))

        delta_v = math.hypot(ego_state[2], ego_state[3]) - ref_state[2]
        return np.array([delta_p, delta_phi, delta_v], dtype=np.float32)