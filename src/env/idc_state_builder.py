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

        # ==========================================
        # 1. 初始化部分 (在类的 __init__ 或 reset 中)
        # ==========================================
        self.expert_pos = None
        self.expert_vel = None
        self.expert_heading = None
        self.EXPERT_TRAJ_LEN = None
        # 预处理专家轨迹数据，转换为适合快速索引的格式
        self._setup_expert_data()
        
        logger.debug(f'专家轨迹长度：{self.EXPERT_TRAJ_LEN}')

    def generate_candidate_paths(self, ego_indices, num_paths=3,
                                  num_points=91, lane_width=3.75):
        """为每个 world 的 controlled agent 生成多条候选 Cubic Bezier 路径"""
        self.candidate_paths = {}
        self.num_candidate_paths = num_paths

        for w in range(self.num_worlds):
            a = ego_indices[w]
            self.candidate_paths[w] = {}

            sp = self.expert_pos[w, a, 0].cpu().numpy()
            ep = self.expert_pos[w, a, -1].cpu().numpy()
            sh = float(self.expert_heading[w, a, 0].item())
            eh = float(self.expert_heading[w, a, -1].item())
            spd = float(self.expert_vel[w, a].norm(dim=-1).mean().item())

            paths = []
            offsets = [0.0]
            if num_paths >= 3:
                offsets = [-lane_width, 0.0, lane_width]

            for offset in offsets:
                path = self._make_bezier_path(
                    sp, sh, ep, eh, offset, spd, num_points)
                paths.append(path)

            self.candidate_paths[w][a] = paths

    def _make_bezier_path(self, p0, h0, p3, h3,
                           lateral_offset, speed, num_points):
        dx = p3[0] - p0[0]
        dy = p3[1] - p0[1]
        dist = float(np.hypot(dx, dy))
        if dist < 1e-6:
            dist = 1.0

        tx, ty = dx / dist, dy / dist
        px, py = -ty, tx

        s = np.array([p0[0] + lateral_offset * px,
                       p0[1] + lateral_offset * py], dtype=np.float32)
        e = np.array([p3[0] + lateral_offset * px,
                       p3[1] + lateral_offset * py], dtype=np.float32)

        d = dist / 3.0
        P0 = s
        P1 = s + np.array([d * np.cos(h0), d * np.sin(h0)], dtype=np.float32)
        P2 = e - np.array([d * np.cos(h3), d * np.sin(h3)], dtype=np.float32)
        P3 = e

        t_vals = np.linspace(0, 1, num_points)
        bt = np.zeros((num_points, 2), dtype=np.float32)
        for i, t in enumerate(t_vals):
            bt[i] = ((1 - t) ** 3 * P0 + 3 * (1 - t) ** 2 * t * P1
                     + 3 * (1 - t) * t ** 2 * P2 + t ** 3 * P3)

        headings = np.zeros(num_points, dtype=np.float32)
        for i in range(num_points - 1):
            headings[i] = np.arctan2(bt[i + 1, 1] - bt[i, 1],
                                     bt[i + 1, 0] - bt[i, 0])
        headings[-1] = headings[-2]

        speeds = np.full(num_points, speed, dtype=np.float32)
        return {'pos': bt, 'heading': headings, 'speed': speeds}

    
    def _setup_expert_data(self):
        # 假设 sim.expert_trajectory_tensor() 返回形状为 [num_worlds, num_agents, 16*91]
        raw_traj = self.sim.expert_trajectory_tensor().to_torch()
        W, A, total_feats = raw_traj.shape
        T = 91  # 根据官方代码确认为 91
        self.EXPERT_TRAJ_LEN = T

        # 按照官方定义的 Offset 一次性切片 (利用 GPU 向量化操作)
        # 形状都会变为 [W, A, T, dim]
        self.expert_pos = raw_traj[:, :, :2*T].reshape(W, A, T, 2)
        self.expert_vel = raw_traj[:, :, 2*T:4*T].reshape(W, A, T, 2)
        self.expert_heading = raw_traj[:, :, 4*T:5*T].reshape(W, A, T, 1)
        
        # 如果以后需要模仿学习，可以顺便存下这个：
        # self.expert_actions = raw_traj[:, :, 6*T:16*T].reshape(W, A, T, 10)

    def reset_world_step(self, world_idx: int, step: int = 0):
        self.step_counter[world_idx] = step

    def increment_step(self, world_idx: int):
        self.step_counter[world_idx] += 1

    # ====================== 主入口 ======================
    def get_idc_observation(self, 
                            world_idx: int, 
                            agent_idx: int,
                            num_other_vehicles: int = 8,
                            path_idx: int = 0) -> np.ndarray:
        s_ego = self.get_ego_state(world_idx, agent_idx)
        s_others = self.get_other_vehicles(world_idx, agent_idx, num_other_vehicles)
        s_ref = self.get_ref_state_from_path(world_idx, agent_idx, path_idx,
                                              ego_x=s_ego[0], ego_y=s_ego[1])
        s_ref_error = self._calc_ref_error(s_ego, s_ref)

        network_state = np.concatenate([
            s_ego,
            s_others.flatten().astype(np.float32),
            s_ref_error.astype(np.float32)
        ])
        return network_state

    def get_idc_observations_batch(self, ego_indices: list,
                                    num_other_vehicles: int = 8,
                                    path_indices: list = None) -> list:
        """
        批量构建所有 world 的网络状态。
        每步只拉一次 GPUDrive tensor，减少 GPU→CPU 传输次数。
        path_indices: 可选，list of int，每个 world 使用的候选路径索引。
        """
        abs_np = self.sim.absolute_self_observation_tensor().to_torch().cpu().numpy()
        rel_np = self.sim.self_observation_tensor().to_torch().cpu().numpy()
        partner_np = self.sim.partner_observations_tensor().to_torch().cpu().numpy()

        states = []
        for w in range(self.num_worlds):
            aidx = ego_indices[w]
            x = float(abs_np[w, aidx, 0])
            y = float(abs_np[w, aidx, 1])
            heading = float(abs_np[w, aidx, 7])
            speed = float(rel_np[w, aidx, 0])
            ego = np.array([x, y, speed, 0.0, heading, 0.0], dtype=np.float32)

            partners = partner_np[w, aidx]
            cos_h = np.cos(heading)
            sin_h = np.sin(heading)
            others_list = []
            for i in range(partners.shape[0]):
                p = partners[i]
                p_speed = p[0]
                rel_x = p[1]
                rel_y = p[2]
                rel_h = p[3]
                abs_h = heading + rel_h
                gx = x + rel_x * cos_h - rel_y * sin_h
                gy = y + rel_x * sin_h + rel_y * cos_h
                dist = np.hypot(gx - x, gy - y)
                others_list.append((dist, [gx, gy, abs_h, p_speed]))
            others_list.sort(key=lambda t: t[0])
            others = np.array([d[1] for d in others_list[:num_other_vehicles]], dtype=np.float32)
            if others.shape[0] < num_other_vehicles:
                pad = np.zeros((num_other_vehicles - others.shape[0], 4), dtype=np.float32)
                others = np.vstack([others, pad]) if others.shape[0] > 0 else pad

            pid = path_indices[w] if path_indices is not None else 0
            _, (rx, ry, rh, rs) = self._nearest_on_candidate(
                w, aidx, pid, x, y)
            ref = np.array([rx, ry, rs, 0.0, rh, 0.0], dtype=np.float32)
            ref_err = self._calc_ref_error(ego, ref)

            state = np.concatenate([
                ego,
                others.flatten().astype(np.float32),
                ref_err.astype(np.float32)
            ])
            states.append(state)
        return states

    def get_ego_positions_batch(self, ego_indices: list) -> np.ndarray:
        """仅拉取 self 绝对观测，返回 [num_worlds, 2] 的 (x, y) 坐标。"""
        abs_np = self.sim.absolute_self_observation_tensor().to_torch().cpu().numpy()
        pos = np.zeros((self.num_worlds, 2), dtype=np.float32)
        for w in range(self.num_worlds):
            a = ego_indices[w]
            pos[w, 0] = float(abs_np[w, a, 0])
            pos[w, 1] = float(abs_np[w, a, 1])
        return pos

    # --------------------------------------------------------
    #  自车状态 [x, y, v_lon, v_lat, heading, omega=0] (车体坐标系)
    #  注意：ego_state 不提供位置和速度分量，只能用 absolute_obs
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
        
        omega = 0.0  # 角速度不直接提供

        return np.array([x, y, speed, 0.0, heading, omega], dtype=np.float32)



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
            
            partner_states.append((distance, [global_x, global_y, abs_heading, speed]))
        
        # 4. 按距离排序
        partner_states.sort(key=lambda x: x[0])
        
        # 5. 截取前 max_partners 个
        if max_partners is not None:
            partner_states = partner_states[:max_partners]
        
        # 6. 转换为 numpy 数组
        result = np.array([state for _, state in partner_states], dtype=np.float32)  # shape (N,4)
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
    

    def get_nearest_ref_point(self, world_idx: int, agent_idx: int, ego_x: float, ego_y: float):
        """
        返回专家轨迹上离自车最近点的索引、位置、航向、速度。
        """
        pos_traj = self.expert_pos[world_idx, agent_idx]  # (T, 2)
        # 计算所有点到自车的欧氏距离平方
        dx = pos_traj[:, 0] - ego_x
        dy = pos_traj[:, 1] - ego_y
        dist_sq = dx*dx + dy*dy
        nearest_idx = torch.argmin(dist_sq).item()
        
        ref_x = pos_traj[nearest_idx, 0].item()
        ref_y = pos_traj[nearest_idx, 1].item()
        ref_heading = self.expert_heading[world_idx, agent_idx, nearest_idx].item()
        ref_speed = torch.norm(self.expert_vel[world_idx, agent_idx, nearest_idx]).item()  # 速度大小
        
        return nearest_idx, (ref_x, ref_y, ref_heading, ref_speed)

    def get_ref_state(self, world_idx: int, agent_idx: int, ego_x: float, ego_y: float) -> np.ndarray:
        """
        基于最近点返回参考状态 [ref_x, ref_y, ref_v_lon, ref_v_lat, ref_heading, 0] (车体坐标系)
        默认使用候选路径 0（中心路径）。
        """
        _, (ref_x, ref_y, ref_heading, ref_speed) = self._nearest_on_candidate(
            world_idx, agent_idx, 0, ego_x, ego_y)
        return np.array([ref_x, ref_y, ref_speed, 0.0, ref_heading, 0.0], dtype=np.float32)

    def _nearest_on_candidate(self, world_idx, agent_idx, path_idx, ego_x, ego_y):
        """在指定候选路径上找最近参考点"""
        path = self.candidate_paths[world_idx][agent_idx][path_idx]
        pos = path['pos']
        dx = pos[:, 0] - ego_x
        dy = pos[:, 1] - ego_y
        dist_sq = dx * dx + dy * dy
        nearest = int(np.argmin(dist_sq))
        return nearest, (float(pos[nearest, 0]), float(pos[nearest, 1]),
                         float(path['heading'][nearest]),
                         float(path['speed'][nearest]))

    def get_ref_state_from_path(self, world_idx, agent_idx, path_idx,
                                  ego_x, ego_y):
        _, (rx, ry, rh, rs) = self._nearest_on_candidate(
            world_idx, agent_idx, path_idx, ego_x, ego_y)
        return np.array([rx, ry, rs, 0.0, rh, 0.0], dtype=np.float32)

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

    def get_ref_states_batch(self, world_indices, ego_xs, ego_ys,
                              ego_indices_map: list,
                              path_indices=None) -> torch.Tensor:
        """
        批量获取参考状态，返回 [batch, 6] tensor。
        path_indices: 可选，list of int，指定每条数据使用的候选路径索引。
        """
        refs = []
        for i in range(len(world_indices)):
            w = world_indices[i]
            a = ego_indices_map[w]
            pid = path_indices[i] if path_indices is not None else 0
            _, (rx, ry, rh, rs) = self._nearest_on_candidate(
                w, a, pid, float(ego_xs[i]), float(ego_ys[i]))
            refs.append([rx, ry, rs, 0.0, rh, 0.0])
        return torch.tensor(refs, device=ego_xs.device, dtype=torch.float32)

    def get_road_edges_batch(self, world_indices, ego_xs, ego_ys) -> torch.Tensor:
        """
        批量获取离自车最近的道路边界点，每步只拉一次 road tensor。
        返回 [batch, 2] tensor。
        """
        road_tensor = self.sim.map_observation_tensor().to_torch().cpu().numpy()
        TYPE_IDX = 6
        ROADLINE_TYPE = 1

        edges = []
        for idx in range(len(world_indices)):
            w = world_indices[idx]
            ex = float(ego_xs[idx])
            ey = float(ego_ys[idx])
            road_points = road_tensor[w]
            line_mask = road_points[:, TYPE_IDX] == ROADLINE_TYPE
            line_points = road_points[line_mask]
            if len(line_points) == 0:
                edges.append([ex, ey])
            else:
                dx = line_points[:, 0] - ex
                dy = line_points[:, 1] - ey
                closest = np.argmin(dx * dx + dy * dy)
                edges.append([float(line_points[closest, 0]),
                              float(line_points[closest, 1])])
        return torch.tensor(edges, device=ego_xs.device, dtype=torch.float32)