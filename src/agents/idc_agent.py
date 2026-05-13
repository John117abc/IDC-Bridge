import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Tuple, List, Dict, Any

# 你的原有模块（确保在 path 中）
from models.actor_critic import DiscreteActor, DiscreteCritic
from models.bicycle import BicycleModel
from buffer import PERBuffer
from utils import DiscreteActionMapper
from utils import get_logger

logger = get_logger('idc-agent')
# ==============================================
# Gumbel-Softmax 辅助函数
# ==============================================
def gumbel_softmax(logits: torch.Tensor, tau: float = 1.0, hard: bool = False, dim: int = -1):
    """
    实现 Gumbel-Softmax 重参数化。
    返回与 logits 同样形状的张量，若不指定 hard 则输出软 one-hot；
    若 hard=True，则返回 one-hot 但梯度与软版本相同。
    """
    gumbels = -torch.empty_like(logits).exponential_().log()  # Gumbel(0,1)
    gumbels = (logits + gumbels) / tau
    y_soft = gumbels.softmax(dim)

    if hard:
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        return (y_hard - y_soft).detach() + y_soft
    return y_soft

# ==============================================
# IDC Agent 适配类（离散动作 + Gumbel-Softmax）
# ==============================================
class DiscreteIDCAgent:
    def __init__(self, env, args, device):
        self.env = env
        self.device = device
        self.num_worlds = env.num_worlds
        self.max_agents = env.max_cont_agents

        # IDC 维度定义
        self.DIM_EGO = 6                  # [x, y, v_lon, v_lat, phi, omega] (omega=0)
        self.DIM_OTHER = 32               # 8 车 × 4 (x, y, phi, v_lon)
        self.DIM_REF_ERROR = 3            # [delta_p, delta_phi, delta_v]
        self.TOTAL_STATE_DIM = self.DIM_EGO + self.DIM_OTHER + self.DIM_REF_ERROR
        self.DIM_ROAD = 80

        # 超参
        self.dt = args.dt
        self.horizon = args.horizon
        self.batch_size = args.batch_size

        # 网络
        self.actor = DiscreteActor(self.TOTAL_STATE_DIM, args.hidden_dim).to(device)
        self.critic = DiscreteCritic(self.TOTAL_STATE_DIM, args.hidden_dim).to(device)
        self.dynamics = BicycleModel(dt=self.dt, L=2.9)   # 自行车模型，假定 omega=0

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.lr_critic)

        # 动作映射器 & Gumbel 温度
        self.action_mapper = DiscreteActionMapper(steer_bins=13, accel_bins=7)
        self.gumbel_tau = 1.0              # 可逐渐退火

        # IDC 成本权重
        self.q_lat = 0.3
        self.q_head = 0.02
        self.q_speed = 0.01
        self.R_matrix = np.diag([0.005, 0.02])

        # GEP 惩罚
        self.init_penalty = args.init_penalty
        self.max_penalty = args.max_penalty
        self.amplifier_c = args.amplifier_c
        self.amplifier_m = args.amplifier_m
        self.gamma = 0.99

        # 安全距离
        self.other_car_min_distance = 0.5   # 米，可根据需要调整
        self.road_min_distance = 0.3
        self.HALF_L = 2.25
        self.HALF_W = 1.0

        # 缓冲区
        self.buffer = PERBuffer(capacity=100000, min_start_train=100)
        self.global_step = 0
        self.gep_iteration = 0

        # 参考速度（可根据任务调整）
        self.ref_vlon = 5.0

        # 道路状态缓存，键为 world_idx
        self.road_states = [None] * self.num_worlds

        # 调试
        self.logger = None  # 可绑定 logging

    # ------------------------------------------------------------
    # 动作选择（推理时的离散采样，不要求可微）
    # ------------------------------------------------------------
    def select_action(self, state: torch.Tensor, deterministic: bool = False):
        """
        state: [batch, TOTAL_STATE_DIM]
        返回: (物理动作 np.array [batch,2], 一维动作索引 np.array [batch])
        """
        state = state.to(self.device)
        with torch.no_grad():
            steer_logits, accel_logits = self.actor(state)

        if deterministic:
            steer_idx = torch.argmax(steer_logits, dim=1)
            accel_idx = torch.argmax(accel_logits, dim=1)
        else:
            steer_dist = torch.distributions.Categorical(logits=steer_logits)
            accel_dist = torch.distributions.Categorical(logits=accel_logits)
            steer_idx = steer_dist.sample()
            accel_idx = accel_dist.sample()

        # 转为物理动作 numpy
        actions_phy = []
        for b in range(state.shape[0]):
            act = self.action_mapper.index_to_action(steer_idx[b].item(), accel_idx[b].item())
            actions_phy.append(act)
        actions_phy = np.stack(actions_phy)

        # 一维动作索引
        action_1d = steer_idx * self.action_mapper.accel_bins + accel_idx
        return actions_phy, action_1d.cpu().numpy()

    # ------------------------------------------------------------
    # 前向推演（使用 Gumbel-Softmax 获得可微动作）
    # ------------------------------------------------------------
    def _get_differentiable_action(self, state_tensor: torch.Tensor):
        """
        输入：当前状态 [B, TOTAL_STATE_DIM]
        输出：可微物理动作 [B, 2]（加速度，转向角），用于前向推演
        """
        steer_logits, accel_logits = self.actor(state_tensor)

        # Gumbel-Softmax 得到软 one-hot
        steer_soft = gumbel_softmax(steer_logits, tau=self.gumbel_tau, hard=False, dim=1)  # [B, steer_bins]
        accel_soft = gumbel_softmax(accel_logits, tau=self.gumbel_tau, hard=False, dim=1)  # [B, accel_bins]

        # 动作值向量（在 GPU 上）
        steer_values = self.action_mapper.steer_edges.to(self.device)   # [steer_bins]
        accel_values = self.action_mapper.accel_edges.to(self.device)   # [accel_bins]

        steer_phy = (steer_soft * steer_values).sum(dim=1, keepdim=True)   # [B, 1]
        accel_phy = (accel_soft * accel_values).sum(dim=1, keepdim=True)   # [B, 1]

        return torch.cat([accel_phy, steer_phy], dim=1)   # [B, 2]

    # ------------------------------------------------------------
    # 参考误差计算（适配 GPUDrive 坐标系）
    # ------------------------------------------------------------
    def _calc_ref_error_from_state(self, ego_state: torch.Tensor,
                                   ref_path_tensor: torch.Tensor) -> torch.Tensor:
        """
        ego_state: [B, 6] (x,y,v_lon,v_lat,phi,omega)
        ref_path_tensor: [B, N, 2] 参考路径点（全局坐标）
        返回: [B, 1, 3] 跟踪误差 (delta_p, delta_phi, delta_v)
        """
        B = ego_state.shape[0]
        if ref_path_tensor.shape[0] == 1 and B > 1:
            ref_path_tensor = ref_path_tensor.repeat(B, 1, 1)

        ego_xy = ego_state[:, :2].unsqueeze(1)    # [B, 1, 2]
        ego_phi = ego_state[:, 4]                 # [B]
        ego_vlon = ego_state[:, 2]                # [B]

        # 最近点索引
        logger.info(f'自车xy：{ego_xy.cpu().detach().numpy()}')
        logger.info(f'自车xy形状：{ego_xy.shape}')
        logger.info(f'参考路径：{ref_path_tensor.cpu().detach().numpy()}')
        logger.info(f'参考形状：{ref_path_tensor.shape}')

        dist = torch.norm(ego_xy - ref_path_tensor.unsqueeze(1), dim=-1)   # [B, 1, N]
        _, closest_idx = torch.min(dist, dim=-1)   # [B, 1]
        closest_idx = closest_idx.squeeze(1)       # [B]

        # 前视距离
        v_lon = ego_vlon
        L_look = torch.clamp(2.0 + 0.5 * v_lon, min=2.0, max=20.0)

        # 获取前视点坐标与航向
        look_xy, look_phi, _ = self._advance_along_path(ref_path_tensor, closest_idx, L_look)

        # 横向误差 (有符号)
        dx = look_xy[:, 0] - ego_state[:, 0]
        dy = look_xy[:, 1] - ego_state[:, 1]
        delta_p = dy * torch.cos(look_phi) - dx * torch.sin(look_phi)

        # 航向误差
        delta_phi = ego_phi - look_phi
        delta_phi = torch.atan2(torch.sin(delta_phi), torch.cos(delta_phi))

        # 速度误差
        delta_v = v_lon - self.ref_vlon

        error = torch.stack([delta_p, delta_phi, delta_v], dim=1).unsqueeze(1)  # [B, 1, 3]
        return error

    # ------------------------------------------------------------
    # 沿路径前进（与 CARLA 版相同）
    # ------------------------------------------------------------
    def _advance_along_path(self, path_xy: torch.Tensor,
                            start_idx: torch.Tensor,
                            dist_forward: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        path_xy: [B, N, 2]
        start_idx: [B]
        dist_forward: [B]
        返回: (look_xy, look_phi, look_idx) 均为 [B]
        """
        if path_xy.dim() == 2:
            path_xy = path_xy.unsqueeze(0)
        B, N, _ = path_xy.shape
        if B == 1 and start_idx.shape[0] > 1:
            path_xy = path_xy.repeat(start_idx.shape[0], 1, 1)
            B = start_idx.shape[0]

        start_idx = start_idx.clamp(0, N-1).long()
        dist_forward = dist_forward.clamp(min=1e-6)

        segs = path_xy[:, 1:] - path_xy[:, :-1]
        seg_len = torch.norm(segs, dim=-1).clamp(min=1e-6)

        cum_from_start = torch.cat([
            torch.zeros(B, 1, device=path_xy.device),
            torch.cumsum(seg_len, dim=1)
        ], dim=1)

        start_dist = cum_from_start[torch.arange(B), start_idx]
        target_dist = start_dist + dist_forward
        total_len = cum_from_start[:, -1]
        overrun = target_dist >= total_len

        look_idx = torch.searchsorted(cum_from_start, target_dist.unsqueeze(1)).squeeze(1)
        look_idx = look_idx.clamp(0, N-1)
        look_idx = torch.where(overrun, torch.full_like(look_idx, N-1), look_idx)

        look_xy = path_xy[torch.arange(B), look_idx]

        next_idx = torch.where(look_idx < N-1, look_idx+1, look_idx)
        prev_idx = torch.where(look_idx > 0, look_idx-1, look_idx)
        dxy = path_xy[torch.arange(B), next_idx] - path_xy[torch.arange(B), look_idx]
        dxy_len = torch.norm(dxy, dim=-1, keepdim=True)
        use_prev = (dxy_len < 1e-6).squeeze(-1)
        dxy = torch.where(
            use_prev.unsqueeze(-1),
            path_xy[torch.arange(B), look_idx] - path_xy[torch.arange(B), prev_idx],
            dxy
        )
        dxy_len = torch.norm(dxy, dim=-1, keepdim=True).clamp(min=1e-6)
        dxy = dxy / dxy_len
        look_phi = torch.atan2(dxy[..., 1], dxy[..., 0])

        return look_xy, look_phi, look_idx

    # ------------------------------------------------------------
    # 前向推演整个时域（核心约束与成本计算）
    # ------------------------------------------------------------
    def _forward_horizon(self, state_tensor: torch.Tensor,
                         ref_path_tensor: torch.Tensor,
                         road_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        state_tensor: [B, TOTAL_STATE_DIM] 当前状态的拼接
        ref_path_tensor: [B, N, 2] 参考路径
        road_state: [B, 80] 道路左右边界（全局坐标）
        返回:
          step_l: [B, horizon] 每步的成本 l
          step_phi: [B, horizon] 每步的约束违反量 phi
          states_traj: [B, horizon, TOTAL_STATE_DIM] 预测状态轨迹
        """
        B = state_tensor.shape[0]
        logger.info(f'road_state的形状：{road_state.shape}')
        logger.info(f'road_state的数据：{road_state.cpu().detach().numpy()}')
        N = road_state.shape[1]
        # 拆分初始状态
        ego_state = state_tensor[:, :self.DIM_EGO].unsqueeze(1)          # [B, 1, 6]
        other_states = state_tensor[:, self.DIM_EGO:self.DIM_EGO+self.DIM_OTHER]
        other_states = other_states.view(B, 1, self.DIM_OTHER//4, 4)     # [B, 1, 8, 4]
        ref_error = state_tensor[:, -self.DIM_REF_ERROR:].unsqueeze(1)   # [B, 1, 3]

        # 道路状态转为左右边界点 [B, N, 20, 2]
        road_left = road_state[:, :self.DIM_ROAD//2].view(B, N, 20, 2, 2)
        road_right = road_state[:, self.DIM_ROAD//2:].view(B, N, 20, 2, 2)

        safe_veh_sq = self.other_car_min_distance ** 2
        safe_road_sq = self.road_min_distance ** 2

        step_l_list = []
        step_phi_list = []
        trajectory_states = []

        current_ego = ego_state
        current_other = other_states
        current_ref_error = ref_error

        for t in range(self.horizon):
            # 构建当前联合状态
            current_state = torch.cat([
                current_ego.view(B, self.DIM_EGO),
                current_other.view(B, self.DIM_OTHER),
                current_ref_error.view(B, self.DIM_REF_ERROR)
            ], dim=1)

            # 可微动作
            phy_action = self._get_differentiable_action(current_state)  # [B, 2]
            accel = phy_action[:, 0:1]
            steer = phy_action[:, 1:2]

            # 动力学推演
            next_ego = self.dynamics(current_ego, phy_action)   # [B, 1, 6]

            # 周车恒速预测
            next_other = self._predict_other(current_other, self.dt)

            # 重新计算参考误差
            next_ref_error = self._calc_ref_error_from_state(next_ego.squeeze(1), ref_path_tensor)

            next_state = torch.cat([
                next_ego.view(B, self.DIM_EGO),
                next_other.view(B, self.DIM_OTHER),
                next_ref_error.view(B, self.DIM_REF_ERROR)
            ], dim=1)
            trajectory_states.append(next_state)

            # ---- 1. 成本 l ----
            lat_err = next_ref_error[:, 0, 0]
            head_err = next_ref_error[:, 0, 1]
            speed_err = next_ref_error[:, 0, 2]
            err_cost = self.q_lat * lat_err**2 + self.q_head * head_err**2 + self.q_speed * speed_err**2

            r_weights = torch.tensor(self.R_matrix.diagonal().copy(), device=self.device).float()
            control_cost = (accel**2 * r_weights[0] + steer**2 * r_weights[1]).squeeze(1)

            step_l = err_cost + control_cost

            # ---- 2. 约束违反 phi ----
            phi_violation = torch.zeros(B, device=self.device)

            # 周车双圆约束
            if self.DIM_OTHER > 0:
                dist_ego = self.HALF_L * 1.0
                ego_cos = torch.cos(next_ego[..., 4])
                ego_sin = torch.sin(next_ego[..., 4])
                ego_x = next_ego[..., 0]
                ego_y = next_ego[..., 1]

                ego_front_x = ego_x + dist_ego * ego_cos
                ego_front_y = ego_y + dist_ego * ego_sin
                ego_rear_x = ego_x - dist_ego * ego_cos
                ego_rear_y = ego_y - dist_ego * ego_sin

                ego_front = torch.stack([ego_front_x, ego_front_y], dim=-1)  # [B,1,2]
                ego_rear  = torch.stack([ego_rear_x, ego_rear_y], dim=-1)

                other_x = next_other[..., 0]
                other_y = next_other[..., 1]
                other_phi = next_other[..., 2]
                other_v = next_other[..., 3]

                other_cos = torch.cos(other_phi)
                other_sin = torch.sin(other_phi)
                dist_other = self.HALF_L * 1.0

                other_front_x = other_x + dist_other * other_cos
                other_front_y = other_y + dist_other * other_sin
                other_rear_x = other_x - dist_other * other_cos
                other_rear_y = other_y - dist_other * other_sin

                other_front = torch.stack([other_front_x, other_front_y], dim=-1)  # [B,1,N,2]
                other_rear  = torch.stack([other_rear_x, other_rear_y], dim=-1)

                ego_front_exp = ego_front.unsqueeze(2)  # [B,1,1,2]
                ego_rear_exp  = ego_rear.unsqueeze(2)

                d_ff = torch.sum((ego_front_exp - other_front)**2, dim=-1)
                d_fr = torch.sum((ego_front_exp - other_rear)**2, dim=-1)
                d_rf = torch.sum((ego_rear_exp  - other_front)**2, dim=-1)
                d_rr = torch.sum((ego_rear_exp  - other_rear)**2, dim=-1)

                min_dist_sq, _ = torch.min(torch.stack([d_ff, d_fr, d_rf, d_rr], dim=-1), dim=-1)

                # 过滤不存在的车
                other_norm = torch.norm(torch.stack([other_x, other_y], dim=-1), dim=-1)
                invalid_mask = other_norm < 1e-3
                min_dist_sq = torch.where(invalid_mask, torch.full_like(min_dist_sq, 1e9), min_dist_sq)

                circle_radius = self.HALF_W * 0.65
                safe_center_dist = 2.0 * circle_radius + self.other_car_min_distance
                safe_center_dist_sq = safe_center_dist ** 2

                veh_violation_sq = torch.clamp(safe_center_dist_sq - min_dist_sq, min=0.0, max=10.0)
                phi_violation += (veh_violation_sq ** 2).sum(dim=[1,2])

            # 道路边界约束
            ego_xy = next_ego[:, :, :2]  # [B,1,2]
            dist_left_sq = torch.sum((ego_xy.unsqueeze(2) - road_left)**2, dim=-1)
            dist_right_sq = torch.sum((ego_xy.unsqueeze(2) - road_right)**2, dim=-1)
            min_left_sq, _ = torch.min(dist_left_sq, dim=-1)
            min_right_sq, _ = torch.min(dist_right_sq, dim=-1)
            left_viol = torch.clamp(safe_road_sq - min_left_sq, min=0.0, max=10.0).squeeze(1)
            right_viol = torch.clamp(safe_road_sq - min_right_sq, min=0.0, max=10.0).squeeze(1)
            phi_violation += (left_viol**2 + right_viol**2)

            step_phi = torch.clamp(phi_violation, max=50.0)

            step_l_list.append(step_l)
            step_phi_list.append(step_phi)

            # 状态更新
            current_ego = next_ego
            current_other = next_other
            current_ref_error = next_ref_error

        step_l = torch.stack(step_l_list, dim=1)     # [B, horizon]
        step_phi = torch.stack(step_phi_list, dim=1)
        states_traj = torch.stack(trajectory_states, dim=1)

        return step_l, step_phi, states_traj

    # ------------------------------------------------------------
    # 周车恒速预测
    # ------------------------------------------------------------
    def _predict_other(self, other_states: torch.Tensor, dt: float) -> torch.Tensor:
        """
        other_states: [B, 1, N, 4]  (x,y,phi,v_lon)
        """
        if other_states.shape[2] == 0:
            return other_states.clone()
        x, y, phi, v = (other_states[..., 0], other_states[..., 1],
                         other_states[..., 2], other_states[..., 3])
        x_next = x + dt * v * torch.cos(phi)
        y_next = y + dt * v * torch.sin(phi)
        return torch.stack([x_next, y_next, phi, v], dim=-1)

    # ------------------------------------------------------------
    # 训练更新（含 GEP）
    # ------------------------------------------------------------
    def update(self):
        if len(self.buffer) < self.batch_size:
            return None

        batch_data = self.buffer.sample_batch(self.batch_size)
        if not batch_data:
            return None

        states_list, road_list, ref_path_list = [], [], []
        for item in batch_data:
            state, action, reward, road_np, done, info = item
            states_list.append(state)
            road_list.append(info['road_state'])       # 从 info 取出
            ref_path_list.append(info['ref_path'])     # 从 info 取出

        state_tensor = torch.tensor(np.stack(states_list), dtype=torch.float32, device=self.device)
        road_tensor = torch.tensor(np.stack(road_list), dtype=torch.float32, device=self.device)
        ref_path_tensor = torch.tensor(np.stack(ref_path_list), dtype=torch.float32, device=self.device)

        # ---- 1. Critic 更新 (策略评估) ----
        with torch.no_grad():
            step_l, step_phi, states_traj = self._forward_horizon(state_tensor, ref_path_tensor, road_tensor)
            # 累计成本（无折扣 gamma=1，可加折扣）
            targets = torch.flip(torch.cumsum(torch.flip(step_l, [1]), dim=1), [1])

        # 构造 Critic 输入：全部预测状态
        all_states = torch.cat([state_tensor.unsqueeze(1), states_traj], dim=1)  # [B, horizon+1, dim]
        critic_inputs = all_states[:, :self.horizon].reshape(-1, self.TOTAL_STATE_DIM)
        critic_targets = targets.reshape(-1, 1)
        pred = self.critic(critic_inputs)
        critic_loss = F.mse_loss(pred, critic_targets)

        self.critic_optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---- 2. Actor 更新 (策略改进) ----
        step_l_actor, step_phi_actor, _ = self._forward_horizon(state_tensor, ref_path_tensor, road_tensor)
        actor_loss = step_l_actor.mean() + self.init_penalty * step_phi_actor.mean()

        self.actor_optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        actor_loss.backward()
        self.actor_optimizer.step()

        self.gep_iteration += 1

        # ---- 3. GEP 惩罚放大 ----
        if self.gep_iteration % self.amplifier_m == 0:
            avg_phi = step_phi_actor.mean().item()
            if avg_phi > 0.5:
                old_penalty = self.init_penalty
                self.init_penalty = min(self.init_penalty * self.amplifier_c, self.max_penalty)
                # 可选日志输出
                # print(f"[GEP] ρ 放大：{old_penalty:.2f} → {self.init_penalty:.2f}, 平均违规={avg_phi:.4f}")

        # 更新 buffer 优先级（基于当前违规量）
        violation_per_sample = step_phi.sum(dim=1).detach().cpu().numpy()
        new_pri = np.maximum(violation_per_sample, 1e-6)
        self.buffer.update_last_batch_priorities(new_pri)

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "penalty": self.init_penalty,
            "gep_iteration": self.gep_iteration,
            "actor_updated": True
        }

    # ------------------------------------------------------------
    # 经验收集（批量）
    # ------------------------------------------------------------
    def collect_experience(self, builder, ego_indices, actions_phy, rewards, dones,ref_raws,roads):
        """
        收集当前步所有 world 的经验到 buffer。
        rewards: [num_worlds, max_agents]
        dones:   [num_worlds, max_agents]
        """
        for w in range(self.num_worlds):
            ego_idx = ego_indices[w]
            # 获取 IDC 观测（已经拼接好的 state、道路等）
            idc_state, road, ref_raw, ref_err, other = builder.get_idc_observation(
                w, ego_idx, perceived_distance=30.0)
            logger.debug(f'idc_state形状:{idc_state.shape}')
            logger.debug(f'road形状:{road.shape}')
            logger.debug(f'ref_raw形状:{ref_raw.shape}')
            logger.debug(f'ref_err形状:{ref_err.shape}')
            logger.debug(f'other形状:{other.shape}')
            action = actions_phy[w]                     # [accel, steer]
            reward = rewards[w, ego_idx].item()
            done   = dones[w, ego_idx].item()

            # 自行构建 info 字典，存储本次需要的路径和道路信息
            info = {
                'ref_path': ref_raws[w],        # 原始参考路径 (N, 2)
                'road_state': roads[w]          # 道路状态 (80,)
            }
            # 压入 buffer（next_obs 填 None，在线更新用不到）
            self.buffer.handle_new_experience((idc_state, action, reward, None, done, info))