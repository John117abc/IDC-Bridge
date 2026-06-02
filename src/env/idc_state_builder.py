import numpy as np
import math
from typing import List, Dict, Tuple, Optional
import torch

from utils import get_logger
logger = get_logger('idc_state_builder')


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
        # 道路张量缓存（一集不变，避免 rollout 中重复 GPU→CPU 传输）
        self._road_cache = None

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
        """为每个 world 生成 3 条候选路径（expert 几何 + 曲率限速）"""
        self.candidate_paths = {}
        self.num_candidate_paths = num_paths

        # 动态获取每个 world 的道路宽度（从 map_observation 的 segment_width）
        road_tensor = self.sim.map_observation_tensor().to_torch().cpu().numpy()
        WIDTH_IDX = 3

        for w in range(self.num_worlds):
            a = ego_indices[w]
            self.candidate_paths[w] = {}

            sp = self.expert_pos[w, a, 0].cpu().numpy()

            # 从道路图中提取该 world 的实际车道宽度
            world_road = road_tensor[w]
            dx = world_road[:, 0] - sp[0]
            dy = world_road[:, 1] - sp[1]
            nearest = int(np.argmin(dx * dx + dy * dy))
            dynamic_width = float(world_road[nearest, WIDTH_IDX])
            if dynamic_width <= 0.0 or not np.isfinite(dynamic_width) or dynamic_width > 15.0:
                dynamic_width = lane_width  # fallback
            logger.debug(f'[LANE-WIDTH] world_{w} width={dynamic_width:.2f}m')

            paths = []
            offsets = [0.0]
            if num_paths >= 3:
                offsets = [-dynamic_width, 0.0, dynamic_width]

            expert_pos_w = self.expert_pos[w, a].cpu().numpy()
            expert_h_w = self.expert_heading[w, a].squeeze(-1).cpu().numpy()

            valid_len = num_points
            for j in range(1, num_points):
                if abs(expert_pos_w[j, 0]) > 5000 or abs(expert_pos_w[j, 1]) > 5000:
                    valid_len = j
                    break
            if valid_len < num_points:
                expert_pos_w[valid_len:, :] = expert_pos_w[valid_len - 1, :]
                expert_h_w[valid_len:] = expert_h_w[valid_len - 1]
                self.expert_pos[w, a] = torch.from_numpy(expert_pos_w)
                self.expert_heading[w, a, :, :] = torch.from_numpy(expert_h_w.reshape(-1, 1))
                logger.debug(f'[TRUNCATE] world_{w} agent_{a} expert truncated at step {valid_len}/{num_points}')

            for offset in offsets:
                if abs(offset) < 1e-6:
                    pos = expert_pos_w.copy()
                else:
                    pos = np.zeros((num_points, 2), dtype=np.float32)
                    for j in range(num_points):
                        h = float(expert_h_w[j])
                        lx = -math.sin(h)
                        ly = math.cos(h)
                        pos[j, 0] = float(expert_pos_w[j, 0]) + offset * lx
                        pos[j, 1] = float(expert_pos_w[j, 1]) + offset * ly

                path = {
                    'pos': pos.astype(np.float32),
                    'heading': expert_h_w.astype(np.float32),
                    'speed': self._curvature_speed(pos),
                }
                road_dists = np.zeros(num_points, dtype=np.float32)
                for j in range(num_points):
                    road_dists[j] = self._road_dist_point(
                        world_road, float(path['pos'][j, 0]), float(path['pos'][j, 1]))
                path['road_dist'] = road_dists
                paths.append(path)

            self.candidate_paths[w][a] = paths

        # 构建 GPU ref_tensor 供 rollout 零 Python 循环索引
        # [num_worlds, 1, num_paths, num_points, 5]: pos_x, pos_y, speed, heading, road_dist
        num_worlds = self.num_worlds
        num_paths = self.num_candidate_paths
        num_points = 91
        ref_data = np.zeros((num_worlds, 1, num_paths, num_points, 5), dtype=np.float32)
        for w in range(num_worlds):
            a = ego_indices[w]
            for p in range(num_paths):
                path = self.candidate_paths[w][a][p]
                n = len(path['pos'])
                ref_data[w, 0, p, :n, 0] = path['pos'][:, 0]
                ref_data[w, 0, p, :n, 1] = path['pos'][:, 1]
                ref_data[w, 0, p, :n, 2] = path['speed'][:]
                ref_data[w, 0, p, :n, 3] = path['heading'][:]
                ref_data[w, 0, p, :n, 4] = path['road_dist'][:]
        self.ref_tensor = torch.from_numpy(ref_data).to(self.expert_pos.device)

        # 诊断: 打印候选路径坐标范围，确认与 ego 坐标系对齐
        for w in range(min(self.num_worlds, 3)):
            a = ego_indices[w]
            for pid, p in enumerate(self.candidate_paths[w][a]):
                pos = p['pos']
                head = p['heading']
                spd = p['speed']
                mid = len(pos) // 2
                logger.info(f'[BEZIER-CHK] world_{w} path_{pid} '
                            f'start=({pos[0,0]:.1f},{pos[0,1]:.1f}) head={head[0]:.2f} spd={spd[0]:.2f}')
                logger.info(f'[BEZIER-CHK] world_{w} path_{pid} '
                            f'mid=({pos[mid,0]:.1f},{pos[mid,1]:.1f}) head={head[mid]:.2f} spd={spd[mid]:.2f}')
                logger.info(f'[BEZIER-CHK] world_{w} path_{pid} '
                            f'end=({pos[-1,0]:.1f},{pos[-1,1]:.1f}) head={head[-1]:.2f} spd={spd[-1]:.2f}')

        # 最近点跳变检测
        self._last_nearest = {}
        self._last_ego = {}

    # ========================================
    # 曲率限速（纯静态，无专家决策信息）
    # ========================================

    @staticmethod
    def _curvature_speed(pos: np.ndarray, v_max: float = 20.0,
                          a_lat_max: float = 2.5) -> np.ndarray:
        """从路径位置的数值曲率计算曲率限速 [num_points]"""
        n = len(pos)
        speeds = np.full(n, v_max, dtype=np.float32)
        for i in range(1, n - 1):
            dx1 = pos[i, 0] - pos[i-1, 0]
            dy1 = pos[i, 1] - pos[i-1, 1]
            dx2 = pos[i+1, 0] - pos[i, 0]
            dy2 = pos[i+1, 1] - pos[i, 1]
            ds1 = float(np.hypot(dx1, dy1)) + 1e-6
            ds2 = float(np.hypot(dx2, dy2)) + 1e-6
            kappa = abs(np.arctan2(dy2, dx2) - np.arctan2(dy1, dx1)) / ((ds1 + ds2) / 2.0)
            speeds[i] = float(np.sqrt(a_lat_max / (kappa + 1e-6)))
        speeds = np.minimum(speeds, v_max)
        speeds[0] = speeds[1]
        speeds[-1] = speeds[-2]
        return speeds

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

    def get_expert_steer_batch(self, world_indices, temporal_indices, ego_indices_map):
        """从 expert pos/heading 数值差分计算近似 steering [batch]"""
        steers = []
        for i in range(len(world_indices)):
            w = world_indices[i]
            a = ego_indices_map[w]
            t = min(int(temporal_indices[i]), self.EXPERT_TRAJ_LEN - 2)
            dh = float(self.expert_heading[w, a, t + 1] - self.expert_heading[w, a, t])
            dx = float(self.expert_pos[w, a, t + 1, 0] - self.expert_pos[w, a, t, 0])
            dy = float(self.expert_pos[w, a, t + 1, 1] - self.expert_pos[w, a, t, 1])
            ds = math.hypot(dx, dy) + 1e-6
            # kinematic bicycle model: δ = atan(L × Δθ / Δs)
            L = 5.0  # wheelbase
            steers.append(math.atan(L * dh / ds))
        return torch.tensor(steers, dtype=torch.float32)

    def clear_cache(self):
        """每集开始时调用，使道路缓存失效。"""
        self._road_cache = None

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
                                    path_indices: list = None,
                                    _abs_np=None, _rel_np=None,
                                    _partner_np=None) -> list:
        """
        批量构建所有 world 的网络状态。
        path_indices: 可选，list of int，每个 world 使用的候选路径索引。
        _abs_np/_rel_np/_partner_np: 可选，预拉取的 tensor（避免重复 GPU→CPU 传输）
        """
        if _abs_np is None:
            abs_np = self.sim.absolute_self_observation_tensor().to_torch().cpu().numpy()
        else:
            abs_np = _abs_np
        if _rel_np is None:
            rel_np = self.sim.self_observation_tensor().to_torch().cpu().numpy()
        else:
            rel_np = _rel_np
        if _partner_np is None:
            partner_np = self.sim.partner_observations_tensor().to_torch().cpu().numpy()
        else:
            partner_np = _partner_np

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
                # 跳过 GPUDrive 空槽位: speed=0, rel_x=0, rel_y=0 的 phantom 车
                if p_speed == 0.0 and rel_x == 0.0 and rel_y == 0.0:
                    continue
                abs_h = heading + rel_h
                gx = x + rel_x * cos_h - rel_y * sin_h
                gy = y + rel_x * sin_h + rel_y * cos_h
                dist = np.hypot(gx - x, gy - y)
                others_list.append((dist, [gx, gy, abs_h, p_speed]))
            others_list.sort(key=lambda t: t[0])
            others = np.array([d[1] for d in others_list[:num_other_vehicles]], dtype=np.float32)
            if others.shape[0] < num_other_vehicles:
                n_pad = num_other_vehicles - others.shape[0]
                pad = np.zeros((n_pad, 4), dtype=np.float32)
                others = np.vstack([others, pad]) if others.shape[0] > 0 else pad
            n_real = min(others.shape[0], num_other_vehicles)
            validity = np.zeros(num_other_vehicles, dtype=np.float32)
            validity[:n_real] = 1.0  # 前 n_real 个是真实车

            pid = path_indices[w] if path_indices is not None else 0
            # 用时序索引替代空间最近点，消除跳变
            t = self.step_counter[w]
            path = self.candidate_paths[w][aidx][pid]
            t = max(0, min(t, len(path['pos']) - 1))
            rx, ry = float(path['pos'][t, 0]), float(path['pos'][t, 1])
            rh, rs = float(path['heading'][t]), float(path['speed'][t])
            ref = np.array([rx, ry, rs, 0.0, rh, 0.0], dtype=np.float32)
            ref_err = self._calc_ref_error(ego, ref)

            # 前瞻参考点（t+5, t+10, t+15），给 Actor 前方弯道/道路信息
            num_pts = len(path['pos'])
            t_l1 = min(t + 5, num_pts - 1)
            t_l2 = min(t + 10, num_pts - 1)
            t_l3 = min(t + 15, num_pts - 1)

            lx1, ly1 = float(path['pos'][t_l1, 0]), float(path['pos'][t_l1, 1])
            lh1 = float(path['heading'][t_l1])
            lat1 = (ly1 - y) * math.cos(lh1) - (lx1 - x) * math.sin(lh1)
            dphi_l1 = lh1 - ego[4]
            dphi_l1 = math.atan2(math.sin(dphi_l1), math.cos(dphi_l1))
            road1 = path['road_dist'][t_l1]
            spd1 = path['speed'][t_l1]

            lx2, ly2 = float(path['pos'][t_l2, 0]), float(path['pos'][t_l2, 1])
            lh2 = float(path['heading'][t_l2])
            lat2 = (ly2 - y) * math.cos(lh2) - (lx2 - x) * math.sin(lh2)
            dphi_l2 = lh2 - ego[4]
            dphi_l2 = math.atan2(math.sin(dphi_l2), math.cos(dphi_l2))
            road2 = path['road_dist'][t_l2]
            spd2 = path['speed'][t_l2]

            lx3, ly3 = float(path['pos'][t_l3, 0]), float(path['pos'][t_l3, 1])
            lh3_ = float(path['heading'][t_l3])
            lat3 = (ly3 - y) * math.cos(lh3_) - (lx3 - x) * math.sin(lh3_)
            dphi_l3 = lh3_ - ego[4]
            dphi_l3 = math.atan2(math.sin(dphi_l3), math.cos(dphi_l3))
            road3 = path['road_dist'][t_l3]
            spd3 = path['speed'][t_l3]

            lookahead_err = np.array([lat1, dphi_l1, road1, spd1,
                                       lat2, dphi_l2, road2, spd2,
                                       lat3, dphi_l3, road3, spd3], dtype=np.float32)
            ref_err = np.concatenate([ref_err, lookahead_err])

            # 诊断大偏离: pos_err > 100m 时打印 ego/ref 坐标和路径索引
            if abs(ref_err[0]) > 100.0:
                logger.debug(f'[LARGE-ERR] world_{w} step={self.step_counter[w]} pos_err={ref_err[0]:.1f}m '
                               f'ego=({x:.1f},{y:.1f}) ref=({rx:.1f},{ry:.1f}) path={pid}')

            state = np.concatenate([
                ego,
                others.flatten().astype(np.float32),
                validity.astype(np.float32),
                ref_err.astype(np.float32),
                np.array([float(t)], dtype=np.float32),
            ])
            states.append(state)
        return states

    def get_ego_positions_batch(self, ego_indices: list,
                                 _abs_np=None) -> np.ndarray:
        """仅拉取 self 绝对观测，返回 [num_worlds, 2] 的 (x, y) 坐标。"""
        if _abs_np is None:
            abs_np = self.sim.absolute_self_observation_tensor().to_torch().cpu().numpy()
        else:
            abs_np = _abs_np
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

    @staticmethod
    def _road_dist_point(road_tensor_world, qx, qy):
        """查询点到最近道路边界点的距离（纯函数，可复用）"""
        mask = road_tensor_world[:, 6] == 1  # TYPE_IDX=6, ROADLINE_TYPE=1
        line_pts = road_tensor_world[mask]
        if len(line_pts) == 0:
            return 999.0
        dx = line_pts[:, 0] - qx
        dy = line_pts[:, 1] - qy
        nearest = int(np.argmin(dx * dx + dy * dy))
        return float(math.hypot(dx[nearest], dy[nearest]))

    def get_road_dist_batch(self, world_indices, temporal_indices,
                             ego_indices_map, path_indices, device) -> torch.Tensor:
        """批量查询预计算的道路边界距离 [batch]"""
        if hasattr(self, 'ref_tensor'):
            w_tensor = torch.tensor(list(world_indices), device=self.ref_tensor.device, dtype=torch.long)
            p_tensor = torch.tensor(list(path_indices), device=self.ref_tensor.device, dtype=torch.long)
            t_tensor = temporal_indices.long().clamp(0, self.ref_tensor.shape[3] - 1).to(self.ref_tensor.device)
            return self.ref_tensor[w_tensor, 0, p_tensor, t_tensor, 4].to(device)
        # CPU 回退
        dists = []
        for i in range(len(world_indices)):
            w = world_indices[i]
            a = ego_indices_map[w]
            pid = path_indices[i]
            t = int(temporal_indices[i])
            path = self.candidate_paths[w][a][pid]
            t = max(0, min(t, len(path['road_dist']) - 1))
            dists.append(float(path['road_dist'][t]))
        return torch.tensor(dists, device=device, dtype=torch.float32)

    @staticmethod
    def _calc_ref_error(ego_state: np.ndarray, ref_state: np.ndarray) -> np.ndarray:
        dx = ref_state[0] - ego_state[0]
        dy = ref_state[1] - ego_state[1]
        delta_p = math.hypot(dx, dy)
        cross = dy * math.cos(ref_state[4]) - dx * math.sin(ref_state[4])
        delta_p *= math.copysign(1.0, cross)

        delta_phi = ref_state[4] - ego_state[4]
        delta_phi = math.atan2(math.sin(delta_phi), math.cos(delta_phi))

        delta_v = math.hypot(ego_state[2], ego_state[3]) - ref_state[2]
        return np.array([delta_p, delta_phi, delta_v], dtype=np.float32)

    def get_ref_states_batch(self, world_indices, ego_xs, ego_ys,
                               ego_indices_map: list,
                               path_indices=None,
                               temporal_indices=None) -> torch.Tensor:
        """
        批量获取参考状态，返回 [batch, 7] tensor。
        列：[rx, ry, rs, 0, rh, 0, nearest_idx]
        当 temporal_indices 传入时，优先走 GPU tensor 索引（零 Python 循环）。
        """
        if temporal_indices is not None and path_indices is not None and hasattr(self, 'ref_tensor'):
            # GPU 快速路径：tensor 索引替代 Python 循环
            w_tensor = torch.tensor(list(world_indices), device=self.ref_tensor.device, dtype=torch.long)
            p_tensor = torch.tensor(list(path_indices), device=self.ref_tensor.device, dtype=torch.long)
            t_tensor = temporal_indices.long().clamp(0, self.ref_tensor.shape[3] - 1).to(self.ref_tensor.device)
            vals = self.ref_tensor[w_tensor, 0, p_tensor, t_tensor]
            rx, ry, rs, rh, rd = vals[:, 0], vals[:, 1], vals[:, 2], vals[:, 3], vals[:, 4]
            return torch.stack([rx, ry, rs, torch.zeros_like(rx), rh, torch.zeros_like(rx), t_tensor.float()], dim=-1)

        # CPU 回退路径
        refs = []
        for i in range(len(world_indices)):
            w = world_indices[i]
            a = ego_indices_map[w]
            pid = path_indices[i] if path_indices is not None else 0
            if temporal_indices is not None:
                t = int(temporal_indices[i])
                path = self.candidate_paths[w][a][pid]
                t = max(0, min(t, len(path['pos']) - 1))
                rx, ry = float(path['pos'][t, 0]), float(path['pos'][t, 1])
                rh, rs = float(path['heading'][t]), float(path['speed'][t])
                nearest = t
            else:
                nearest, (rx, ry, rh, rs) = self._nearest_on_candidate(
                    w, a, pid, float(ego_xs[i]), float(ego_ys[i]))
            refs.append([rx, ry, rs, 0.0, rh, 0.0, float(nearest)])
        return torch.tensor(refs, device=ego_xs.device, dtype=torch.float32)

    def get_road_edges_batch(self, world_indices, ego_xs, ego_ys) -> torch.Tensor:
        """
        批量获取离自车最近的道路边界点，每步只拉一次 road tensor。
        返回 [batch, 2] tensor。
        """
        if self._road_cache is None:
            self._road_cache = self.sim.map_observation_tensor().to_torch().cpu().numpy()
        road_tensor = self._road_cache
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