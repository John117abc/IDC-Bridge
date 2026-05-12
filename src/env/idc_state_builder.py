import numpy as np
import math
from typing import List, Dict, Tuple, Optional
import torch

class GPUDriveObservationBuilder:
    """
    将 GPUDrive 环境状态转换为 IDC 所需的观察向量。
    依赖底层 madrona_gpudrive 的 SimManager 或 GPUDriveTorchEnv，
    以及场景 JSON 数据（用于提取道路边缘和目标点）。
    """

    def __init__(self, sim_or_env, scene_json_list: List[Dict]):
        """
        :param sim_or_env: GPUDriveTorchEnv 或 madrona_gpudrive.SimManager
        :param scene_json_list: 每个 world 的场景 JSON 数据列表，顺序需与 sim 一致
        """
        # 统一获取底层 sim 对象
        if hasattr(sim_or_env, 'sim'):
            self.sim = sim_or_env.sim
        else:
            self.sim = sim_or_env

        self.scene_data = scene_json_list
        self.num_worlds = len(scene_json_list)

        # 步数计数器（外部需手动 reset / increment）
        self.step_counter = {w: 0 for w in range(self.num_worlds)}

        # ---------- 缓存道路边缘点 ----------
        # edge_points[idx] 包含该世界所有道路边缘点的 (x, y) 列表
        self.edge_points = []
        for scene in self.scene_data:
            points = []
            for road in scene.get('roads', []):
                # 收集 road_edge 和 road_line 类型的几何点
                if road.get('type') in ('road_edge', 'road_line'):
                    geom = road.get('geometry', [])
                    for g in geom:
                        points.append((g['x'], g['y']))
            self.edge_points.append(points)

        # ---------- 缓存每个 agent 的 goal 位置 ----------
        self.goal_positions = []
        for scene in self.scene_data:
            goals = {}
            for obj in scene.get('objects', []):
                aid = obj.get('id', -1)
                if aid != -1 and 'goalPosition' in obj:
                    gp = obj['goalPosition']
                    goals[aid] = (gp['x'], gp['y'])
            self.goal_positions.append(goals)

        # ---------- 尝试获取专家轨迹张量 ----------
        # 形状 [worlds, agents, steps, features]
        self.expert_traj = None
        self.EXPERT_TRAJ_LEN = None
        if hasattr(self.sim, 'expert_trajectory_tensor'):
            try:
                self.expert_traj = self.sim.expert_trajectory_tensor().to_torch()
                self.EXPERT_TRAJ_LEN  = self.expert_traj.shape[2] // 16
                if self.expert_traj is not None:
                    print("expert_traj shape:", self.expert_traj.shape)
                    print("expert_traj[0,0,:5]:", self.expert_traj[0, 0, :5].cpu().numpy())

            except Exception as e:
                print(f"Warning: unable to get expert_trajectory_tensor: {e}")
                self.expert_traj = None

        # ---------- 尝试获取绝对自身观测张量（备用） ----------
        self.abs_obs = None
        if hasattr(self.sim, 'absolute_self_observation_tensor'):
            try:
                self.abs_obs = self.sim.absolute_self_observation_tensor().to_torch()
            except Exception as e:
                print(f"Warning: unable to get absolute_self_observation_tensor: {e}")
                self.abs_obs = None

    def reset_world_step(self, world_idx: int, step: int = 0):
        """重置某世界的当前步数"""
        self.step_counter[world_idx] = step

    def increment_step(self, world_idx: int):
        """步进某世界的步数计数器"""
        self.step_counter[world_idx] += 1

    # ========================================================
    #  主入口：生成 IDC 全量观测
    # ========================================================
    def get_idc_observation(self, world_idx: int, agent_idx: int,
                            perceived_distance: float = 30.0,
                            num_other_vehicles: int = 8,
                            num_road_points: int = 20) -> Tuple[np.ndarray, ...]:
        """
        返回:
            network_state (41,): [ego(6) + other(32) + ref_error(3)]
            s_road (80,): 全局坐标系下的道路边缘点
            s_ref_raw (6,): 参考路径原始状态
            s_ref_error (3,): 参考误差
            s_other (8,4): 周车状态
        """
        cur_step = self.step_counter[world_idx]

        s_ego = self._get_ego_state(world_idx, agent_idx, cur_step)
        s_other = self._get_other_vehicles(world_idx, agent_idx, cur_step,
                                           distance_threshold=perceived_distance,
                                           max_num=num_other_vehicles)
        s_road = self._get_road_edges(world_idx, agent_idx, s_ego,
                                      num_points=num_road_points,
                                      front_distance=perceived_distance)
        s_ref_raw = self._get_ref_state(world_idx, agent_idx, cur_step, s_ego,
                                        default_speed=10.0, num_path_points=50)
        s_ref_error = self._calc_ref_error(s_ego, s_ref_raw)

        network_state = np.concatenate([s_ego, s_other.flatten().astype(np.float32),
                                        s_ref_error.astype(np.float32)])
        return network_state, s_road, s_ref_raw, s_ref_error, s_other

    # --------------------------------------------------------
    #  1. 自车状态 [x, y, v_lon, v_lat, φ, ω]
    # --------------------------------------------------------
    def _get_ego_state(self, world_idx: int, agent_idx: int, step: int) -> np.ndarray:
        """
        优先使用专家轨迹中的位置、速度、航向；
        若不可用则回退到 absolute_self_observation_tensor。
        """
        # 尝试从专家轨迹获取
        if self.expert_traj is not None and step < self.EXPERT_TRAJ_LEN:
            pos_all, vel_all, heading_all = self._get_expert_arrays(world_idx)
            x, y =pos_all[agent_idx, step, 0].cpu().numpy(), pos_all[agent_idx, step, 1].cpu().numpy()
            vel = vel_all[agent_idx, step].cpu().numpy()  # (2,)
            heading = heading_all[agent_idx, step].item()
            vx, vy = vel[0], vel[1]
            omega = 0.0
        else:
            # 回退到绝对观测张量
            if self.abs_obs is None:
                raise RuntimeError("Neither expert_trajectory nor absolute_self_observation available.")
            state = self.abs_obs[world_idx, agent_idx].cpu().numpy()
            x = state[0]
            y = state[1]
            vx = state[2] if len(state) > 2 else 0.0
            vy = state[3] if len(state) > 3 else 0.0
            heading = state[4] if len(state) > 4 else 0.0
            omega = state[5] if len(state) > 5 else 0.0

        # 计算车辆坐标系下的纵向/横向速度
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        v_lon = vx * cos_h + vy * sin_h
        v_lat = -vx * sin_h + vy * cos_h

        return np.array([x, y, v_lon, v_lat, heading, omega], dtype=np.float32)

    # --------------------------------------------------------
    #  2. 周车状态 (8,4) [x, y, φ, v_lon]
    # --------------------------------------------------------
    def _get_other_vehicles(self, world_idx: int, ego_idx: int, step: int,
                            distance_threshold: float, max_num: int) -> np.ndarray:
        """
        获取距离自车最近的 max_num 辆周车，状态为 (x, y, φ, v_lon)
        """
        # 若无任何数据源，返回全零
        if self.expert_traj is None and self.abs_obs is None:
            return np.zeros((max_num, 4), dtype=np.float32)

        # 优先使用专家轨迹（方案A）
        if self.expert_traj is not None and step < self.EXPERT_TRAJ_LEN:
            pos_all, vel_all, heading_all = self._get_expert_arrays(world_idx)
            ego_x = pos_all[ego_idx, step, 0].item()
            ego_y = pos_all[ego_idx, step, 1].item()
            other_list = []
            num_agents = pos_all.shape[0]
            for other_idx in range(num_agents):
                if other_idx == ego_idx:
                    continue
                ox = pos_all[other_idx, step, 0].item()
                oy = pos_all[other_idx, step, 1].item()
                ovx = vel_all[other_idx, step, 0].item()
                ovy = vel_all[other_idx, step, 1].item()
                oheading = heading_all[other_idx, step].item()
                dx = ox - ego_x
                dy = oy - ego_y
                dist = math.hypot(dx, dy)
                if dist > distance_threshold or dist < 1e-3:
                    continue
                v_lon = ovx * math.cos(oheading) + ovy * math.sin(oheading)
                other_list.append((dist, [ox, oy, oheading, v_lon]))
            other_list.sort(key=lambda v: v[0])
            result = [v[1] for v in other_list[:max_num]]
            while len(result) < max_num:
                result.append([0.0, 0.0, 0.0, 0.0])
            return np.array(result, dtype=np.float32)

        # 回退：使用 absolute_self_observation_tensor
        if self.abs_obs is not None:
            ego_state = self._get_ego_state(world_idx, ego_idx, step)  # 回退获取自车坐标
            ego_x, ego_y = ego_state[0], ego_state[1]
            controlled = self.sim.controlled_state_tensor().to_torch()
            valid_mask = controlled[world_idx].squeeze(-1) > 0
            other_list = []
            num_agents = self.abs_obs.shape[1]
            for other_idx in range(num_agents):
                if other_idx == ego_idx or not valid_mask[other_idx]:
                    continue
                s = self.abs_obs[world_idx, other_idx].cpu().numpy()
                ox, oy = s[0], s[1]
                ovx = s[2] if len(s) > 2 else 0.0
                ovy = s[3] if len(s) > 3 else 0.0
                oheading = s[7] if len(s) > 7 else 0.0
                dx = ox - ego_x
                dy = oy - ego_y
                dist = math.hypot(dx, dy)
                if dist > distance_threshold or dist < 1e-3:
                    continue
                v_lon = ovx * math.cos(oheading) + ovy * math.sin(oheading)
                other_list.append((dist, [ox, oy, oheading, v_lon]))
            other_list.sort(key=lambda v: v[0])
            result = [v[1] for v in other_list[:max_num]]
            while len(result) < max_num:
                result.append([0.0, 0.0, 0.0, 0.0])
            return np.array(result, dtype=np.float32)

        return np.zeros((max_num, 4), dtype=np.float32)

    def _get_expert_arrays(self, world_idx: int):
        """返回 (pos, vel, heading) 三个数组，形状分别为 (num_agents, T, 2), (num_agents, T, 2), (num_agents, T)"""
        traj = self.expert_traj[world_idx]  # [num_agents, T*16]
        num_agents, T16 = traj.shape
        T = T16 // 16
        pos = traj[:, :2 * T].reshape(num_agents, T, 2)  # (num_agents, T, 2)
        vel = traj[:, 2 * T:4 * T].reshape(num_agents, T, 2)  # (num_agents, T, 2)
        heading = traj[:, 4 * T:5 * T].reshape(num_agents, T)  # (num_agents, T)
        return pos, vel, heading

    # --------------------------------------------------------
    #  3. 道路边缘观测 (全局坐标，80维)
    # --------------------------------------------------------
    def _get_road_edges(self, world_idx: int, agent_idx: int, ego_state: np.ndarray,
                        num_points: int, front_distance: float) -> np.ndarray:
        """
        返回前方 num_points 对左右边缘点（全局坐标），
        排列：左1x,左1y, ..., 左Nx,左Ny, 右1x,右1y, ..., 右Nx,右Ny
        """
        ego_x, ego_y = ego_state[0], ego_state[1]
        ego_heading = ego_state[4]
        cos_h = math.cos(ego_heading)
        sin_h = math.sin(ego_heading)

        all_pts = self.edge_points[world_idx]
        if not all_pts:
            return np.zeros(num_points * 4, dtype=np.float32)

        left_candidates = []
        right_candidates = []

        for (px, py) in all_pts:
            dx = px - ego_x
            dy = py - ego_y
            proj_long = dx * cos_h + dy * sin_h   # 纵向投影
            if proj_long < 0 or proj_long > front_distance:
                continue
            proj_lat = -dx * sin_h + dy * cos_h   # 横向投影
            if proj_lat > 0:
                left_candidates.append((proj_long, proj_lat, px, py))
            else:
                right_candidates.append((proj_long, -proj_lat, px, py))

        # 按纵向距离排序，取前 num_points
        left_candidates.sort(key=lambda v: v[0])
        right_candidates.sort(key=lambda v: v[0])
        left_sel = left_candidates[:num_points]
        right_sel = right_candidates[:num_points]

        def flatten(pts):
            flat = []
            for _, _, x, y in pts:
                flat.extend([x, y])
            return flat

        left_flat = flatten(left_sel)
        right_flat = flatten(right_sel)

        # 补齐到 num_points*2
        left_flat += [0.0] * (num_points * 2 - len(left_flat))
        right_flat += [0.0] * (num_points * 2 - len(right_flat))

        return np.array(left_flat + right_flat, dtype=np.float32)

    # --------------------------------------------------------
    #  4. 参考路径原始状态 (基于专家轨迹)
    # --------------------------------------------------------
    def _get_ref_state(self, world_idx: int, agent_idx: int, step: int,
                       ego_state: np.ndarray, default_speed: float = 10.0,
                       num_path_points: int = 50) -> np.ndarray:
        """
        从当前步开始，取后续轨迹点作为参考路径，取第一个参考点及方向。
        :return: [x_ref, y_ref, v_ref, 0, φ_ref, 0]
        """
        if self.expert_traj is not None and step < self.EXPERT_TRAJ_LEN:
            pos_all, vel_all, heading_all = self._get_expert_arrays(world_idx)
            traj_pos = pos_all[agent_idx]  # (T,2)
            traj_vel = vel_all[agent_idx]  # (T,2)
            # 取当前步及下一个点
            ref_idx = step
            next_idx = min(step + 1, self.EXPERT_TRAJ_LEN - 1)
            ref_x, ref_y = traj_pos[ref_idx, 0].item(), traj_pos[ref_idx, 1].item()
            dx = traj_pos[next_idx, 0].item() - ref_x
            dy = traj_pos[next_idx, 1].item() - ref_y
            ref_yaw = math.atan2(dy, dx) if math.hypot(dx, dy) > 1e-6 else ego_state[4]
            vx, vy = traj_vel[ref_idx, 0].item(), traj_vel[ref_idx, 1].item()
            ref_speed = math.hypot(vx, vy) if (vx or vy) else default_speed
        else:
            # 无专家轨迹，退化为目标点方向
            goals = self.goal_positions[world_idx]
            gx, gy = goals.get(agent_idx, (ego_state[0], ego_state[1]))
            ref_x, ref_y = gx, gy
            dx = ref_x - ego_state[0]
            dy = ref_y - ego_state[1]
            ref_yaw = math.atan2(dy, dx) if (abs(dx) > 1e-3 or abs(dy) > 1e-3) else ego_state[4]
            ref_speed = default_speed

        return np.array([ref_x, ref_y, ref_speed, 0.0, ref_yaw, 0.0], dtype=np.float32)

    # --------------------------------------------------------
    #  5. 参考误差
    # --------------------------------------------------------
    @staticmethod
    def _calc_ref_error(ego_state: np.ndarray, ref_state: np.ndarray) -> np.ndarray:
        dx = ref_state[0] - ego_state[0]
        dy = ref_state[1] - ego_state[1]
        delta_p = math.hypot(dx, dy)
        cross = dy * math.cos(ref_state[4]) - dx * math.sin(ref_state[4])
        delta_p *= math.copysign(1.0, cross)

        delta_phi = ego_state[4] - ref_state[4]
        delta_phi = math.atan2(math.sin(delta_phi), math.cos(delta_phi))

        delta_v = ego_state[2] - ref_state[2]

        return np.array([delta_p, delta_phi, delta_v], dtype=np.float32)