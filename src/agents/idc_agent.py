import math
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Tuple, List, Dict, Any

from models.continuous_actor_critic import ContinuousActor, ContinuousCritic
from models.kinematic_bicycle import KinematicBicycleModel
from buffer import PERBuffer
from utils import get_logger
from utils import save_checkpoint, load_checkpoint

logger = get_logger('idc-agent')


# ==============================================
# IDC Agent
# ==============================================
class DiscreteIDCAgent:
    def __init__(self, env, args, device,state_builder,ego_indices):
        self.args = args
        self.env = env
        self.device = device
        self.num_worlds = env.num_worlds
        self.max_agents = env.max_cont_agents
        self.state_builder = state_builder  # 传入状态构建器实例
        self.ego_indices = ego_indices  # 每个世界中自车的索引列表

        # IDC 维度定义
        self.DIM_EGO = 6                  # [x, y, v_lon, v_lat, phi, omega] (车体坐标系)
        self.DIM_OTHERS = 32               # 8 车 × 4 (x, y, phi, v_lon)
        self.DIM_VALIDITY = 8             # 8 车 validity mask: 1=真实车 0=padding
        self.DIM_REF_ERROR = 12           # [dp, dphi, dv, lat+lphi+lroad ×3 for t+3, t+6, t+9]
        self.DIM_TEMPORAL = 1             # 时序索引，避免空间最近点跳变
        self.TOTAL_STATE_DIM = self.DIM_EGO + self.DIM_OTHERS + self.DIM_VALIDITY + self.DIM_REF_ERROR + self.DIM_TEMPORAL
        self.DIM_ROAD = 80

        # 超参
        self.dt = args.dt
        self.horizon = args.horizon
        self.batch_size = args.batch_size

        # 网络
        self.actor = ContinuousActor(self.TOTAL_STATE_DIM, args.hidden_dim).to(device)
        self.critic = ContinuousCritic(self.TOTAL_STATE_DIM, args.hidden_dim).to(device)
        self.dynamics = KinematicBicycleModel(dt=self.dt)

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.lr_critic)

        # IDC 成本权重
        self.pos_err_weight = 0.3
        self.speed_err_weight = 0.04
        self.heading_err_weight = 0.3
        self.steer_cost_weight = 0.3
        self.acc_cost_weight = 0.005
        self.lookahead_pos_weight = 0.05
        self.lookahead_heading_weight = 0.05

        # GEP 惩罚
        self.rho = args.init_penalty
        self.max_penalty = args.max_penalty
        self.amplifier_c = args.amplifier_c
        self.gamma = 0.99
        self.pev_step = 0
        self.pim_interval = args.pim_interval

        # 安全距离
        self.D_veh_safe = 2.0    # 两车圆心威胁距离: 并排邻车道 ~3.75m 不受罚
        self.D_road_safe = 2.0
        self.HALF_L = 2.25
        self.HALF_W = 1.0

        # 缓冲区
        self.buffer = PERBuffer(capacity=100000, min_start_train=args.batch_size)
        self.global_step = 1
        self.gep_iteration = 0

        # 道路状态缓存，键为 world_idx
        self.road_states = [None] * self.num_worlds

        # 回合数
        # 探索噪声
        self.noise_std = 0.1
        self.noise_decay_rate = 0.98
        self.noise_std_min = 0.03

        # 诊断开关
        self.fix_speed = getattr(args, 'fix_speed', False)
        self.fix_heading = getattr(args, 'fix_heading', False)
        self.no_sign = getattr(args, 'no_sign', False)

        # 回合数
        self.globe_eps = 0

        # 历史损失
        self.history_loss = []


    def select_action(self, state, deterministic=False):
        """
        state: list of state objects (batch)
        返回 tensor [batch, max_agents, 2]，每个动作为 [delta, a]
        """
        with torch.no_grad():
            state_tensor = self.batch_state_to_tensor(state)        # [batch, TOTAL_STATE_DIM]
            logger.debug(f'选择动作: state_tensor shape={state_tensor.shape}')
            
            norm_action = self.actor(state_tensor)                  # [batch, max_agents * 2]
            batch_size = state_tensor.shape[0]
            # 假设 self.max_agents 已定义
            norm_action = norm_action.view(batch_size, self.max_agents, 2)   # [batch, max_agents, 2]

            if not deterministic:
                # 生成与 norm_action 相同形状的高斯噪声，标准差可设为 0.1（也可按维度分别设置）
                noise = torch.normal(0, self.noise_std, size=norm_action.shape, device=norm_action.device)
                norm_action = norm_action + noise
                norm_action = torch.clamp(norm_action, -1.0, 1.0)

            # 转向映射：[-1,1] → [-0.6,0.6] rad
            delta_phy = norm_action[..., 0] * 0.6

            # 加速度映射：分段线性，norm=0 时输出 0（默认滑行），范围保持 [-3.0, 1.5]
            a_phy = torch.where(
                norm_action[..., 1] >= 0,
                norm_action[..., 1] * 1.5,
                norm_action[..., 1] * 3.0
            )

            # 组合最终动作 [batch, max_agents, 2]
            action = torch.stack([a_phy, delta_phy], dim=-1)

            # [DIAG] 前几个 world 的实际动作值和 norm_action 原始输出
            if self.global_step % 10 == 0:
                n_show = min(20, batch_size)
                for i in range(n_show):
                    raw_steer = norm_action[i, 0, 0].item()  # tanh 前 [-1,1]
                    raw_acc = norm_action[i, 0, 1].item()
                    logger.debug(f'[DIAG-act] world_{i} acc={action[i,0,0].item():.3f} steer={action[i,0,1].item():.3f} '
                                f'| norm_steer={raw_steer:.4f} norm_acc={raw_acc:.4f}')
            return action


    def predict_others(self,others_state):
        """
        others_state: [batch, N * 4]  (px, py, phi, speed)
        返回下一时刻的 others_state
        """
        logger.debug(f'预测周车下一状态: others_state shape={others_state.shape}')
        px, py, phi, speed = others_state.view(others_state.shape[0], int(self.DIM_OTHERS / 4), 4).unbind(dim=-1)  # [batch, N]
        px_next = px + speed * torch.cos(phi) * self.dt
        py_next = py + speed * torch.sin(phi) * self.dt
        phi_next = phi
        speed_next = speed
        return torch.stack([px_next, py_next, phi_next, speed_next], dim=-1)
    
    def compute_rollout_target(self, states, world_indices, path_indices=None):
        total_utility = torch.zeros(states.shape[0], device=self.device)
        s, w_i, p_i = states, world_indices, path_indices
        l_min_log, l_max_log = float('inf'), float('-inf')
        for t in range(self.horizon):
            u = self.actor(s)
            if torch.isnan(u).any():
                logger.warning(f'[NaN] actor output at rollout step {t}')
            s = self.f_pred_batch(s, u, w_i, p_i)
            if torch.isnan(s).any():
                logger.warning(f'[NaN] f_pred state at rollout step {t}')
            l = self.utility_batch(s, u, w_i, p_i)
            if torch.isnan(l).any():
                logger.warning(f'[NaN] utility at rollout step {t}')
            l_min_log = min(l_min_log, l.min().item())
            l_max_log = max(l_max_log, l.max().item())
            total_utility = total_utility + (self.gamma ** t) * torch.clamp(l, -50.0, 50.0)
        if self.global_step % 50 == 0:
            tu = total_utility
            logger.debug(f'[DIAG-critic] utility raw range=[{l_min_log:.2f}, {l_max_log:.2f}], '
                        f'target raw mean={tu.mean().item():.2f} min={tu.min().item():.2f} max={tu.max().item():.2f}')
        return total_utility

    def batch_state_to_tensor(self, states):
        logger.debug(f'将状态对象列表转换为张量表示: batch_size={len(states)},第一个状态={states[0] if len(states) > 0 else "N/A"}')
        # 一次性将所有状态转为张量
        states_tensors = [torch.from_numpy(s) for s in states]   # list of [TOTAL_STATE_DIM]
        
        ego_tensors = torch.stack([s[:self.DIM_EGO] for s in states_tensors]).to(self.device)  # [batch, DIM_EGO]
        others_tensors = torch.stack([s[self.DIM_EGO:self.DIM_EGO+self.DIM_OTHERS] for s in states_tensors]).to(self.device)
        validity_tensors = torch.stack([s[self.DIM_EGO+self.DIM_OTHERS:self.DIM_EGO+self.DIM_OTHERS+self.DIM_VALIDITY] for s in states_tensors]).to(self.device)
        ref_error_tensors = torch.stack([s[self.DIM_EGO+self.DIM_OTHERS+self.DIM_VALIDITY:self.DIM_EGO+self.DIM_OTHERS+self.DIM_VALIDITY+self.DIM_REF_ERROR] for s in states_tensors]).to(self.device)
        temporal_tensors = torch.stack([s[self.DIM_EGO+self.DIM_OTHERS+self.DIM_VALIDITY+self.DIM_REF_ERROR:
                                          self.DIM_EGO+self.DIM_OTHERS+self.DIM_VALIDITY+self.DIM_REF_ERROR+self.DIM_TEMPORAL] for s in states_tensors]).to(self.device)
        
        others_flat = others_tensors.view(len(states), -1)
        state_tensor = torch.cat([ego_tensors, others_flat, validity_tensors, ref_error_tensors, temporal_tensors], dim=-1)
        return state_tensor
    
    def f_pred_batch(self, states, actions, w_i, p_i=None):
        """
        states: tensor [batch, TOTAL_STATE_DIM]
        actions: tensor [batch, 2]
        w_i: list/tuple of world indices [batch]
        p_i: optional list of path indices [batch]
        """
        ego_tensors = states[:, :self.DIM_EGO]
        others_tensors = states[:, self.DIM_EGO:self.DIM_EGO+self.DIM_OTHERS]
        # 提取 validity mask 并在推演中保持不变
        val_start = self.DIM_EGO + self.DIM_OTHERS
        validity_tensors = states[:, val_start:val_start + self.DIM_VALIDITY]
        # 提取时序索引并累加（模拟时间推进）
        temp_start = self.DIM_EGO + self.DIM_OTHERS + self.DIM_VALIDITY + self.DIM_REF_ERROR
        temporal_idx = states[:, temp_start:temp_start + self.DIM_TEMPORAL].squeeze(-1).long()
        temporal_next = (temporal_idx + 1).long()
        
        # 1. 动力学推演（将Actor原始输出映射为物理量，与select_action保持一致）
        delta_phy = actions[..., 0] * 0.6  # 转向：[-1,1] → [-0.6, 0.6] rad
        a_phy = torch.where(actions[..., 1] >= 0,
                            actions[..., 1] * 1.5,
                            actions[..., 1] * 3.0)  # 加速度：[-1,1] → [-3.0, 1.5] m/s²
        actions_phy = torch.stack([a_phy, delta_phy], dim=-1)  # [acc, steer]
        ego_next = self.dynamics(ego_tensors[...,:6], actions_phy)
        x_raw, y_raw, theta_next, v_next = ego_next[:, 0], ego_next[:, 1], ego_next[:, 2], ego_next[:, 3]

        # 2. 参考点查询（用时序索引替代空间最近点，消除跳变）
        refs = self.state_builder.get_ref_states_batch(
            w_i, x_raw.detach(), y_raw.detach(), self.ego_indices, p_i,
            temporal_indices=temporal_next)
        ref_x, ref_y = refs[:, 0], refs[:, 1]

        # 3. 直接使用动力学输出（坏world已由训练脚本过滤，不再需要clamp）
        x_next = x_raw
        y_next = y_raw

        # 4. 组装 next_state
        ego_next_formatted = torch.stack([
            x_next, y_next, 
            v_next, torch.zeros_like(v_next),  # v_lon, v_lat
            theta_next, torch.zeros_like(theta_next) # phi, omega
        ], dim=-1)
        
        others_next = self.predict_others(others_tensors)
        
        # 5. 参考误差（在动力学输出上计算）
        dx = ref_x - x_next
        dy = ref_y - y_next
        delta_p = torch.hypot(dx, dy)
        if not self.no_sign:
            cross = dy * torch.cos(refs[:, 4]) - dx * torch.sin(refs[:, 4])
            sign = torch.where(cross >= 0, torch.tensor(1.0, device=self.device),
                               torch.tensor(-1.0, device=self.device))
            delta_p = delta_p * sign

        # 诊断：clamp 后仍大偏离
        if (delta_p.abs() > 100.0).any():
            big = (delta_p.abs() > 100.0).nonzero(as_tuple=True)[0].tolist()
            for i in big[:3]:
                logger.debug(f'[FPRED-ERR] sample_{i} delta_p={delta_p[i].item():.1f}m '
                               f'ego=({x_next[i].item():.1f},{y_next[i].item():.1f}) '
                               f'ref=({ref_x[i].item():.1f},{ref_y[i].item():.1f})')

        delta_phi = refs[:, 4] - theta_next
        delta_phi = torch.atan2(torch.sin(delta_phi), torch.cos(delta_phi))
        
        ego_speed = torch.hypot(ego_next_formatted[:, 2], ego_next_formatted[:, 3])
        delta_v = ego_speed - refs[:, 2]

        # 6. 前瞻参考点：给 Actor 前方弯道/道路信息（t+3, t+6, t+9）
        tl1 = torch.clamp(temporal_next + 3, max=89)
        tl2 = torch.clamp(temporal_next + 6, max=89)
        tl3 = torch.clamp(temporal_next + 9, max=89)
        refs_l1 = self.state_builder.get_ref_states_batch(
            w_i, x_raw.detach(), y_raw.detach(), self.ego_indices, p_i,
            temporal_indices=tl1)
        refs_l2 = self.state_builder.get_ref_states_batch(
            w_i, x_raw.detach(), y_raw.detach(), self.ego_indices, p_i,
            temporal_indices=tl2)
        refs_l3 = self.state_builder.get_ref_states_batch(
            w_i, x_raw.detach(), y_raw.detach(), self.ego_indices, p_i,
            temporal_indices=tl3)

        lat_l1 = (refs_l1[:, 1] - y_next) * torch.cos(refs_l1[:, 4]) - (refs_l1[:, 0] - x_next) * torch.sin(refs_l1[:, 4])
        dphi_l1 = refs_l1[:, 4] - theta_next
        dphi_l1 = torch.atan2(torch.sin(dphi_l1), torch.cos(dphi_l1))
        road_l1 = self.state_builder.get_road_dist_batch(w_i, tl1, self.ego_indices, p_i, x_next.device)

        lat_l2 = (refs_l2[:, 1] - y_next) * torch.cos(refs_l2[:, 4]) - (refs_l2[:, 0] - x_next) * torch.sin(refs_l2[:, 4])
        dphi_l2 = refs_l2[:, 4] - theta_next
        dphi_l2 = torch.atan2(torch.sin(dphi_l2), torch.cos(dphi_l2))
        road_l2 = self.state_builder.get_road_dist_batch(w_i, tl2, self.ego_indices, p_i, x_next.device)

        lat_l3 = (refs_l3[:, 1] - y_next) * torch.cos(refs_l3[:, 4]) - (refs_l3[:, 0] - x_next) * torch.sin(refs_l3[:, 4])
        dphi_l3 = refs_l3[:, 4] - theta_next
        dphi_l3 = torch.atan2(torch.sin(dphi_l3), torch.cos(dphi_l3))
        road_l3 = self.state_builder.get_road_dist_batch(w_i, tl3, self.ego_indices, p_i, x_next.device)

        ref_error_tensors = torch.stack([delta_p, delta_phi, delta_v,
                                          lat_l1, dphi_l1, road_l1,
                                          lat_l2, dphi_l2, road_l2,
                                          lat_l3, dphi_l3, road_l3], dim=-1)
        
        # 组合并返回最终状态
        next_states = []
        for i in range(len(states)):
            next_s = self.tensor_to_state_tensor(ego_next_formatted[i], others_next[i], validity_tensors[i], ref_error_tensors[i], temporal_next[i])
            next_states.append(next_s)
            
        return torch.stack(next_states)
    
    def tensor_to_state_tensor(self, ego_tensor, others_tensor, validity_tensor, ref_error_tensor, temporal_idx):
        """组合状态张量"""
        return torch.cat([ego_tensor, others_tensor.view(-1), validity_tensor, ref_error_tensor, temporal_idx.view(-1)], dim=-1)

    def update_critic(self, states, world_indices, path_indices=None):
        logger.debug(f'更新 Critic: states batch size={states.shape[0]}')
        with torch.no_grad():
            targets = self.compute_rollout_target(states, world_indices, path_indices).detach()
        logger.debug(f"[DEBUG] target stats: min={targets.min().item():.2f}, max={targets.max().item():.2f}, mean={targets.mean().item():.2f}, has_nan={torch.isnan(targets).any()}")

        if torch.isnan(targets).any():
            logger.warning(f'Target 包含 NaN，跳过本次 Critic 更新')
            return float('nan')

        values = self.critic(states)
        logger.debug(f"[DEBUG] value stats: min={values.min().item():.2f}, max={values.max().item():.2f}, mean={values.mean().item():.2f}, has_nan={torch.isnan(values).any()}")

        loss = F.mse_loss(values, targets.unsqueeze(1))
        logger.debug(f"[DEBUG] loss = {loss.item():.4f}")

        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        # 打印梯度范数
        total_norm = 0
        for p in self.critic.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        logger.debug(f"[DEBUG] grad norm = {total_norm:.4f}")

        self.critic_optimizer.step()
        return loss.detach().item()

    def update_actor(self, states, w_i, p_i=None):
        total_cost = torch.zeros(states.shape[0], device=self.device)
        s = states
        ref_start = self.DIM_EGO + self.DIM_OTHERS + self.DIM_VALIDITY
        for t in range(self.horizon):
            u = self.actor(s)
            s = self.f_pred_batch(s, u, w_i, p_i)
            l = self.utility_batch(s, u, w_i, p_i)
            if t == 0:
                logger.info(f'[LA-DIAG] l1(lat={s[0, ref_start+3]:.2f} dphi={s[0, ref_start+4]:.3f} road={s[0, ref_start+5]:.1f}) '
                            f'l2(lat={s[0, ref_start+6]:.2f} dphi={s[0, ref_start+7]:.3f} road={s[0, ref_start+8]:.1f}) '
                            f'l3(lat={s[0, ref_start+9]:.2f} dphi={s[0, ref_start+10]:.3f} road={s[0, ref_start+11]:.1f}) '
                            f'delta_p={s[0, ref_start]:.2f} delta_phi={s[0, ref_start+1]:.3f}')
            l = torch.clamp(l, -5000.0, 5000.0)
            p = self.penalty_batch(s, w_i)
            if t == 0:
                logger.debug(f'[DIAG-pen] raw penalty min={p.min().item():.2f} max={p.max().item():.2f} '
                            f'mean={p.mean().item():.2f}  (rho={self.rho:.4f})')
            total_cost = total_cost + (self.gamma ** t) * (l + self.rho * p)

        actor_loss = total_cost.mean()

        # 衰减探索噪声（所有模式统一衰减）
        old_std = self.noise_std
        self.noise_std = max(self.noise_std_min, self.noise_std * self.noise_decay_rate)
        if abs(old_std - self.noise_std) > 1e-4:
            logger.debug(f'[DIAG-noise] decay {old_std:.4f} -> {self.noise_std:.4f}')

        # 保存权重副本，用于 NaN 回滚
        backup = {name: param.clone() for name, param in self.actor.named_parameters()}

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # 检测 NaN 权重并回滚
        has_nan = False
        for name, param in self.actor.named_parameters():
            if torch.isnan(param).any():
                has_nan = True
                break
        if has_nan:
            logger.warning(f'Actor 权重出现 NaN，回滚到更新前。rho={self.rho:.4f}, gep_iter={self.gep_iteration}')
            with torch.no_grad():
                for name, param in self.actor.named_parameters():
                    param.copy_(backup[name])
            self.pev_step = max(0, self.pev_step - 1)
            self.gep_iteration = max(0, self.gep_iteration - 1)
            self.rho = self.rho / self.amplifier_c
            actor_loss = torch.tensor(float('nan'))

        return actor_loss.detach().item() if not has_nan else float('nan')

    def utility_batch(self, s, u, w_i, p_i=None):
        """
        s: [batch, TOTAL_STATE_DIM], u: [batch, 2]
        """
        ego = s[:, :self.DIM_EGO]
        # 用时序索引替代空间最近点查表
        temp_start = self.DIM_EGO + self.DIM_OTHERS + self.DIM_VALIDITY + self.DIM_REF_ERROR
        temporal_idx = s[:, temp_start:temp_start + self.DIM_TEMPORAL].squeeze(-1).long()
        refs = self.state_builder.get_ref_states_batch(
            w_i, ego[:, 0], ego[:, 1], self.ego_indices, p_i,
            temporal_indices=temporal_idx)

        pos_err = torch.sqrt((ego[:, 0] - refs[:, 0]) ** 2
                            + (ego[:, 1] - refs[:, 1]) ** 2 + 1e-8)
        # 直接读 state 中的 delta_phi（与 f_pred_batch 和 _calc_ref_error 含义一致）
        ref_start = self.DIM_EGO + self.DIM_OTHERS + self.DIM_VALIDITY
        heading_err = s[:, ref_start + 1]
        speed_err = torch.sqrt((ego[:, 2] - refs[:, 2]) ** 2
                              + (ego[:, 3] - refs[:, 3]) ** 2 + 1e-8)

        steer_cost = u[:, 0] ** 2
        acc_cost = u[:, 1] ** 2

        l = (self.pos_err_weight * pos_err ** 2
             + (0.0 if self.fix_speed else self.speed_err_weight) * speed_err ** 2
             + (0.0 if self.fix_heading else self.heading_err_weight) * heading_err ** 2
             + self.steer_cost_weight * steer_cost
             + self.acc_cost_weight * acc_cost
             + self.lookahead_pos_weight * (s[:, ref_start + 3] ** 2 + s[:, ref_start + 6] ** 2 + s[:, ref_start + 9] ** 2)
             + self.lookahead_heading_weight * (s[:, ref_start + 4] ** 2 + s[:, ref_start + 7] ** 2 + s[:, ref_start + 10] ** 2))

        # 诊断：每次调用自动抓 pos_err 最大的样本
        max_idx = pos_err.argmax().item()
        if pos_err[max_idx] > 5.0:
            logger.debug(f'[TRACK-DIAG] worst sample_{max_idx} world_{w_i[max_idx]} '
                        f'ego=({ego[max_idx,0]:.1f},{ego[max_idx,1]:.1f},θ={ego[max_idx,4]:.2f}) '
                        f'ref=({refs[max_idx,0]:.1f},{refs[max_idx,1]:.1f},θ={refs[max_idx,4]:.2f}) '
                        f'nearest={int(refs[max_idx,6].item())} pos={pos_err[max_idx]:.1f} dhead={heading_err[max_idx]:.2f}')
        return l

    def _two_circle_centers(self, x, y, phi):
        """
        x, y, phi: [batch, ...] tensors
        返回前圆心和后圆心: (front, rear) 各 [batch, ..., 2]
        """
        dx = self.HALF_L * torch.cos(phi)
        dy = self.HALF_L * torch.sin(phi)
        front = torch.stack([x + dx, y + dy], dim=-1)
        rear = torch.stack([x - dx, y - dy], dim=-1)
        return front, rear

    def penalty_batch(self, s, w_i):
        """
        s: [batch, TOTAL_STATE_DIM]
        返回 [batch] 标量惩罚值（双圆碰撞模型，可导）
        """
        ego = s[:, :self.DIM_EGO]
        others = s[:, self.DIM_EGO:self.DIM_EGO + self.DIM_OTHERS].view(-1, 8, 4)
        val_start = self.DIM_EGO + self.DIM_OTHERS
        validity = s[:, val_start:val_start + self.DIM_VALIDITY]  # [batch, 8]

        # 自车双圆心 [batch, 2, 2]
        ego_front, ego_rear = self._two_circle_centers(ego[:, 0], ego[:, 1], ego[:, 4])
        ego_circles = torch.stack([ego_front, ego_rear], dim=1)  # [batch, 2, 2]

        # 周车双圆心 [batch, 8, 2, 2]
        oth_front, oth_rear = self._two_circle_centers(
            others[:, :, 0], others[:, :, 1], others[:, :, 2])
        oth_circles = torch.stack([oth_front, oth_rear], dim=2)  # [batch, 8, 2, 2]

        # 4 组距离: ego[2] × other[8,2] → [batch, 8, 2, 2]
        diff = ego_circles[:, None, :, None, :] - oth_circles[:, :, None, :, :]
        dist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)  # [batch, 8, 2, 2]
        # 每辆周车取最危险的一对圆碰撞，用 validity mask 屏蔽 padding 车
        pair_pen = F.relu(self.D_veh_safe - dist).pow(2)          # [batch, 8, 2, 2]
        per_veh_max = pair_pen.flatten(2).max(dim=2).values       # [batch, 8]
        veh_violation = (per_veh_max * validity).sum(dim=1)       # [batch] — padding 车不计入

        # 道路边界惩罚：线性，始终有梯度拉回道路
        edge_pts = self.state_builder.get_road_edges_batch(
            w_i, ego[:, 0], ego[:, 1])
        edge_dist = torch.sqrt((edge_pts[:, 0] - ego[:, 0]) ** 2
                              + (edge_pts[:, 1] - ego[:, 1]) ** 2 + 1e-8)
        road_violation = F.relu(edge_dist - self.D_road_safe)  # 线性，不平方

        return veh_violation + road_violation
    
    def update(self):
        samples = self.buffer.sample_batch(self.batch_size)
        states, word_indexs, path_indices = zip(*samples)
        states_tensor = self.batch_state_to_tensor(states)

        # PEV: 每次 update 都更新 critic
        critic_loss = self.update_critic(states_tensor, word_indexs, path_indices)
        self.pev_step += 1

        # PIM: 积累足够 PEV 步后才更新 actor 并放大 ρ
        actor_loss = None
        if self.pev_step >= self.pim_interval:
            self.pev_step = 0
            self.rho = min(self.rho * self.amplifier_c, self.max_penalty)
            self.gep_iteration += 1
            actor_loss = self.update_actor(states_tensor, word_indexs, path_indices)

        return critic_loss, actor_loss
        
    def update_ego_indices(self, new_ego_indices):
        self.ego_indices = new_ego_indices

    def clear_buffer(self):
        self.buffer = PERBuffer(capacity=100000, min_start_train=self.args.batch_size)

    def save(self, save_info: Dict[str, Any]) -> None:
        # 满足回合数才会保存模型
        if self.globe_eps % self.args.save_freq == 0:
            logger.info(f'保存模型: globe_eps={self.globe_eps}, global_step={self.global_step}')
            model = {'actor': self.actor, 'critic': self.critic}
            optimizer = {'actor_optim': self.actor_optimizer, 'critic_optim': self.critic_optimizer}
            extra_info = {
                'config': self.args,
                'global_step': self.global_step,
                'history': self.history_loss,
                'globe_eps': self.globe_eps,
                'state_dim': self.TOTAL_STATE_DIM,
                'rho': self.rho,
                'gep_iteration': self.gep_iteration,
                
            }
            metrics = {'episode': extra_info['globe_eps']}
            save_checkpoint(model=model, 
                            model_name='idc-waymo-v1.0', 
                            optimizer=optimizer,
                            file_dir=self.args.file_dir,
                            env_name=save_info.get('env_name', 'unknown_env'),
                            extra_info=extra_info, 
                            metrics=metrics)
    

    def load(self, path: str) -> Dict[str, Any]:
        checkpoint = load_checkpoint(
            model={'actor': self.actor, 'critic': self.critic},
            filepath=path,
            optimizer={'actor_optim': self.actor_optimizer, 'critic_optim': self.critic_optimizer},
            device=self.device
        )
        loaded_dim = checkpoint.get('state_dim', self.TOTAL_STATE_DIM)
        if loaded_dim != self.TOTAL_STATE_DIM:
            logger.warning(f"加载模型维度{loaded_dim}与当前{self.TOTAL_STATE_DIM}不一致")
        self.globe_eps = checkpoint['globe_eps']
        self.history_loss = checkpoint['history']
        self.global_step = checkpoint['global_step']
        self.rho = checkpoint['rho'] == 0.0 and self.args.init_penalty or checkpoint['rho']
        self.gep_iteration = checkpoint['gep_iteration']
        return checkpoint
        