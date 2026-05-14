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
    def get_idc_observation(self, 
                            world_idx: int, 
                            agent_idx: int,
                            num_other_vehicles: int = 8) -> Tuple[np.ndarray, ...]:
        """
        返回:
            network_state: (6 + num_other_vehicles*6 + num_road_points*6*2 + 3)
            s_road: (num_road_points*2*6,) 左右道路边缘展平向量
            s_ref_raw: (6,)
            s_ref_error: (3,)
            s_other: (num_other_vehicles, 6)
        """
        s_ego = self.get_ego_state(world_idx, agent_idx)
        s_others = self.get_other_vehicles(world_idx, agent_idx, num_other_vehicles)
        s_road = self.get_road_edges(world_idx, ego_x=s_ego[0], ego_y=s_ego[1])
        s_ref = self.get_ref_state(world_idx, agent_idx)
        s_ref_error = self._calc_ref_error(s_ego, s_ref)

        # 周车只取[x,y,heading,vy],目前获得的值是这样的[global_x, global_y, vx, vy, abs_heading, 0.0]
        s_idc_others = s_others[:, [0, 1, 4, 3]]  # 选择 x, y, heading, vy 四个维度
        network_state = np.concatenate([
            s_ego,
            s_idc_others.flatten().astype(np.float32),
            s_ref_error.astype(np.float32)
        ])

        # raw_state
        raw_state = np.concatenate([
            s_ego,
            s_others.flatten().astype(np.float32),
            s_ref.astype(np.float32),
            s_road.flatten().astype(np.float32),
        ])
        return network_state, raw_state

    # --------------------------------------------------------
    #  自车状态 [x, y, vx, vy, heading, omega=0]
    #  注意：ego_state 不提供位置和分速度，只能用 absolute_obs
    # --------------------------------------------------------
    def get_ego_state(self, world_idx: int, agent_idx: int) -> np.ndarray:
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



    def get_other_vehicles(self, 
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
        ego_state = self.get_ego_state(world_idx, ego_agent_idx)  # [x, y, vx, vy, heading, omega]
        ego_x, ego_y = ego_state[0], ego_state[1]
        ego_heading = ego_state[4]
        
        # 2. 获取周车观测张量 (num_worlds, max_agents, max_agents-1, 9)
        partner_tensor = self.sim.partner_observations_tensor().to_torch()
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

    def get_road_edges(self, world_idx: int, ego_x: float, ego_y: float) -> np.ndarray:
        """
        返回离自车最近的 RoadLine 点（全局绝对坐标）。
        如果没有任何 RoadLine 点，返回自车坐标。
        返回格式: [x, y, 0, 0, 0, 0]
        """
        # 获取全局道路观测张量 (num_worlds, num_road_points, 9)
        road_tensor = self.sim.map_observation_tensor().to_torch()
        # 取出指定世界的所有道路点 (num_road_points, 9)
        road_points = road_tensor[world_idx].cpu().numpy()
        
        # 筛选类型为 RoadLine 的点 (type 索引为 6，值为 1)
        TYPE_IDX = 6
        ROADLINE_TYPE = 1
        line_points = road_points[road_points[:, TYPE_IDX] == ROADLINE_TYPE]
        
        if len(line_points) == 0:
            # 没有 RoadLine 点，返回自车坐标
            return np.array([ego_x, ego_y, 0., 0., 0., 0.], dtype=np.float32)
        
        # 计算每个点到自车的欧氏距离
        dx = line_points[:, 0] - ego_x
        dy = line_points[:, 1] - ego_y
        distances = np.hypot(dx, dy)
        closest_idx = np.argmin(distances)
        closest_x = line_points[closest_idx, 0]
        closest_y = line_points[closest_idx, 1]
        
        return np.array([closest_x, closest_y, 0., 0., 0., 0.], dtype=np.float32)
    
    # --------------------------------------------------------
    #  参考路径
    # --------------------------------------------------------
    def get_ref_state(self, world_idx: int, agent_idx: int) -> np.ndarray:
        """
        专家参考状态：[x_ref, y_ref, vx_ref, 0, heading_ref, 0]
        1. 优先使用专家轨迹的未来点；
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