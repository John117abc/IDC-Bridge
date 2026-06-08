import math
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Tuple, List, Dict, Any
from collections import deque

from models.continuous_actor_critic import ContinuousActor, ContinuousCritic
from models.kinematic_bicycle import KinematicBicycleModel
from buffer import PERBuffer
from utils import get_logger
from utils import save_checkpoint, load_checkpoint

logger = get_logger('idc-agent')


# ==============================================
# IDC Agent (Baseline — 原论文方法)
# MLP Actor + 经典 GEP 累加 + 无辅助 loss
# ==============================================
class DiscreteIDCAgentBaseline:
    def __init__(self, env, config, device, state_builder, ego_indices):
        self.config = config
        self.env = env
        self.device = device
        self.num_worlds = env.num_worlds
        self.max_agents = env.max_cont_agents
        self.state_builder = state_builder
        self.ego_indices = ego_indices

        # IDC 维度定义 (基线: 仅 dp/dphi/dv)
        self.DIM_EGO = 6
        self.DIM_OTHERS = 32
        self.DIM_VALIDITY = 8
        self.DIM_REF_ERROR = 15           # 与 state builder 对齐 (dp/dphi/dv + 12 lookahead)
        self.DIM_TEMPORAL = 1
        self.TOTAL_STATE_DIM = self.DIM_EGO + self.DIM_OTHERS + self.DIM_VALIDITY + self.DIM_REF_ERROR + self.DIM_TEMPORAL

        # 超参
        self.dt = config.dt
        self.horizon = config.horizon
        self.batch_size = config.batch_size

        # 网络 (MLP)
        self.actor = ContinuousActor(self.TOTAL_STATE_DIM, config.hidden_dim).to(device)
        self.critic = ContinuousCritic(self.TOTAL_STATE_DIM, config.hidden_dim).to(device)
        self.dynamics = KinematicBicycleModel(
            dt=self.dt, L=config.wheelbase, lr_ratio=config.lr_ratio, v_max=config.v_max)

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.lr_actor_max)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.lr_critic_max)

        # 余弦退火调度器
        steps_per_epoch = 91
        buffer_fill = max(1, self.batch_size // self.num_worlds)
        pev_per_epoch = max(1, steps_per_epoch - buffer_fill)
        total_u_steps = pev_per_epoch * config.epochs
        self.actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.actor_optimizer, T_max=total_u_steps, eta_min=config.lr_actor_min)
        self.critic_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.critic_optimizer, T_max=total_u_steps, eta_min=config.lr_critic_min)

        # IDC 成本权重 (标准 Q/R)
        self.pos_err_weight = config.pos_err_weight
        self.speed_err_weight = config.speed_err_weight
        self.heading_err_weight = config.heading_err_weight
        self.steer_cost_weight = config.steer_cost_weight
        self.acc_cost_weight = config.acc_cost_weight

        # 经典 GEP 惩罚 (累加器)
        self.rho = config.init_penalty
        self.max_penalty = config.max_penalty
        self.amplifier_c = config.amplifier_c
        self.gamma = config.gamma
        self.pev_step = 0
        self.pim_interval = config.pim_interval

        # 安全距离
        self.D_veh_safe = config.D_veh_safe
        self.D_road_safe = config.D_road_safe
        self.HALF_L = config.half_length
        self.HALF_W = config.half_width

        # 缓冲区 (单帧，无窗口)
        self.buffer = PERBuffer(
            capacity=config.buffer_capacity,
            min_start_train=config.batch_size,
            window_size=1,
            state_dim=self.TOTAL_STATE_DIM,
        )
        self.global_step = 1
        self.gep_iteration = 0

        # 探索噪声
        self.noise_std = config.noise_std
        self.noise_decay_rate = config.noise_decay_rate
        self.noise_std_min = config.noise_std_min

        # 诊断开关
        self.fix_speed = getattr(config, 'fix_speed', False)
        self.fix_heading = getattr(config, 'fix_heading', False)
        self.no_sign = getattr(config, 'no_sign', False)

        self.globe_eps = 0
        self.history_loss = []
        self.epoch_history_pdms = []

    def select_action(self, state, deterministic=False):
        """state: list of numpy arrays [w][TOTAL_STATE_DIM]"""
        with torch.no_grad():
            state_tensor = torch.from_numpy(np.stack(state)).to(self.device)
            raw_output = self.actor(state_tensor)
            batch_size = raw_output.shape[0]
            raw_output = raw_output.view(batch_size, self.max_agents, 2)

            if not deterministic:
                noise = torch.normal(0, self.noise_std * 0.3, size=raw_output.shape,
                                     device=raw_output.device)
                raw_output = raw_output + noise

            delta_phy = torch.clamp(raw_output[..., 0] * 0.3, -0.6, 0.6)
            acc_norm = torch.tanh(raw_output[..., 1])
            a_phy = torch.where(acc_norm >= 0, acc_norm * 1.5, acc_norm * 3.0)
            return torch.stack([a_phy, delta_phy], dim=-1)

    def predict_others(self, others_state):
        """恒速预测 (原论文方法)"""
        px, py, phi, speed = others_state.view(others_state.shape[0], int(self.DIM_OTHERS / 4), 4).unbind(dim=-1)
        px_next = px + speed * torch.cos(phi) * self.dt
        py_next = py + speed * torch.sin(phi) * self.dt
        return torch.stack([px_next, py_next, phi, speed], dim=-1)

    def compute_rollout_target(self, states, world_indices, path_indices=None):
        total_utility = torch.zeros(states.shape[0], device=self.device)
        s = states
        for t in range(self.horizon):
            u = self.actor(s)
            s = self.f_pred_batch(s, u, world_indices, path_indices)
            total_utility = total_utility + (self.gamma ** t) * torch.clamp(
                self.utility_batch(s, u, world_indices, path_indices), -50.0, 50.0)
        return total_utility

    def f_pred_batch(self, states, actions, w_i, p_i=None):
        """单帧推理：动力学推演 → 组装下一状态"""
        ego_tensors = states[:, :self.DIM_EGO]
        others_tensors = states[:, self.DIM_EGO:self.DIM_EGO + self.DIM_OTHERS]
        val_start = self.DIM_EGO + self.DIM_OTHERS
        validity_tensors = states[:, val_start:val_start + self.DIM_VALIDITY]
        temp_start = self.DIM_EGO + self.DIM_OTHERS + self.DIM_VALIDITY + self.DIM_REF_ERROR
        temporal_idx = states[:, temp_start:temp_start + self.DIM_TEMPORAL].squeeze(-1).long()
        temporal_next = (temporal_idx + 1).long()

        delta_phy = torch.clamp(actions[..., 0] * 0.3, -0.6, 0.6)
        acc_norm = torch.tanh(actions[..., 1])
        a_phy = torch.where(acc_norm >= 0, acc_norm * 1.5, acc_norm * 3.0)
        ego_next = self.dynamics(ego_tensors[..., :6], torch.stack([a_phy, delta_phy], dim=-1))
        x_raw, y_raw, theta_next, v_next = ego_next[:, 0], ego_next[:, 1], ego_next[:, 2], ego_next[:, 3]

        refs = self.state_builder.get_ref_states_batch(
            w_i, x_raw.detach(), y_raw.detach(), self.ego_indices, p_i, temporal_indices=temporal_next)
        ref_x, ref_y = refs[:, 0], refs[:, 1]

        ego_next_formatted = torch.stack([
            x_raw, y_raw, v_next, torch.zeros_like(v_next),
            theta_next, torch.zeros_like(theta_next)], dim=-1)

        others_next = self.predict_others(others_tensors)

        dx = ref_x - x_raw; dy = ref_y - y_raw
        delta_p = torch.hypot(dx, dy)
        if not self.no_sign:
            cross = dy * torch.cos(refs[:, 4]) - dx * torch.sin(refs[:, 4])
            sign = torch.where(cross >= 0, torch.tensor(1.0, device=self.device),
                               torch.tensor(-1.0, device=self.device))
            delta_p = delta_p * sign

        delta_phi = torch.atan2(torch.sin(refs[:, 4] - theta_next), torch.cos(refs[:, 4] - theta_next))
        delta_v = torch.hypot(ego_next_formatted[:, 2], ego_next_formatted[:, 3]) - refs[:, 2]

        # t+5 横向误差 (道路 penalty 需要)
        tl1 = torch.clamp(temporal_next + 5, max=91)
        refs_l1 = self.state_builder.get_ref_states_batch(
            w_i, x_raw.detach(), y_raw.detach(), self.ego_indices, p_i, temporal_indices=tl1)
        lat_l1 = (refs_l1[:, 1] - x_raw) * torch.cos(refs_l1[:, 4]) - (refs_l1[:, 0] - x_raw) * torch.sin(refs_l1[:, 4])

        ref_error_tensors = F.pad(
            torch.stack([delta_p, delta_phi, delta_v, lat_l1], dim=-1),
            (0, self.DIM_REF_ERROR - 4))

        return torch.cat([ego_next_formatted, others_next.view(len(states), -1),
                          validity_tensors, ref_error_tensors,
                          temporal_next.unsqueeze(-1).float()], dim=-1)

    def update_critic(self, states, world_indices, path_indices=None):
        with torch.no_grad():
            targets = self.compute_rollout_target(states, world_indices, path_indices).detach()
        if torch.isnan(targets).any():
            return float('nan')
        values = self.critic(states)
        loss = F.mse_loss(values, targets.unsqueeze(1))
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        self.critic_scheduler.step()
        return loss.detach().item()

    def update_actor(self, states, w_i, p_i=None):
        total_cost = torch.zeros(states.shape[0], device=self.device)
        s = states
        max_penalty = 0.0
        for t in range(self.horizon):
            u = self.actor(s)
            s = self.f_pred_batch(s, u, w_i, p_i)
            l = self.utility_batch(s, u, w_i, p_i)
            l = torch.clamp(l, -5000.0, 5000.0)
            p = self.penalty_batch(s, w_i, p_i)
            p = torch.clamp(p, max=100.0)
            max_penalty = max(max_penalty, p.max().item())
            total_cost = total_cost + (self.gamma ** t) * (l + self.rho * p)

        actor_loss = total_cost.mean()
        has_violation = max_penalty > 0.5

        self.noise_std = max(self.noise_std_min, self.noise_std * self.noise_decay_rate)

        backup = {name: param.clone() for name, param in self.actor.named_parameters()}
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        self.actor_scheduler.step()

        has_nan = any(torch.isnan(param).any() for param in self.actor.parameters())
        if has_nan:
            with torch.no_grad():
                for name, param in self.actor.named_parameters():
                    param.copy_(backup[name])
            self.pev_step = max(0, self.pev_step - 1)
            self.gep_iteration = max(0, self.gep_iteration - 1)
            self.rho = max(0.0, self.rho - self.amplifier_c)
            actor_loss = float('nan')
            has_violation = False
            max_penalty = 0.0

        return actor_loss.detach().item() if not has_nan else float('nan'), has_violation, max_penalty

    def utility_batch(self, s, u, w_i, p_i=None):
        ego = s[:, :self.DIM_EGO]
        temp_start = self.DIM_EGO + self.DIM_OTHERS + self.DIM_VALIDITY + self.DIM_REF_ERROR
        temporal_idx = s[:, temp_start:temp_start + self.DIM_TEMPORAL].squeeze(-1).long()
        refs = self.state_builder.get_ref_states_batch(
            w_i, ego[:, 0], ego[:, 1], self.ego_indices, p_i, temporal_indices=temporal_idx)

        pos_err = torch.sqrt((ego[:, 0] - refs[:, 0]) ** 2 + (ego[:, 1] - refs[:, 1]) ** 2 + 1e-8)
        ref_start = self.DIM_EGO + self.DIM_OTHERS + self.DIM_VALIDITY
        heading_err = s[:, ref_start + 1]
        speed_err = torch.sqrt((ego[:, 2] - refs[:, 2]) ** 2 + (ego[:, 3] - refs[:, 3]) ** 2 + 1e-8)

        return (self.pos_err_weight * pos_err ** 2
                + self.speed_err_weight * speed_err ** 2
                + self.heading_err_weight * heading_err ** 2
                + self.steer_cost_weight * u[:, 0] ** 2
                + self.acc_cost_weight * u[:, 1] ** 2)

    def _two_circle_centers(self, x, y, phi):
        dx = self.HALF_L * torch.cos(phi); dy = self.HALF_L * torch.sin(phi)
        return torch.stack([x + dx, y + dy], dim=-1), torch.stack([x - dx, y - dy], dim=-1)

    def penalty_batch(self, s, w_i, p_i=None):
        ego = s[:, :self.DIM_EGO]
        others = s[:, self.DIM_EGO:self.DIM_EGO + self.DIM_OTHERS].view(-1, 8, 4)
        val_start = self.DIM_EGO + self.DIM_OTHERS
        validity = s[:, val_start:val_start + self.DIM_VALIDITY]

        ego_front, ego_rear = self._two_circle_centers(ego[:, 0], ego[:, 1], ego[:, 4])
        ego_circles = torch.stack([ego_front, ego_rear], dim=1)
        oth_front, oth_rear = self._two_circle_centers(others[:, :, 0], others[:, :, 1], others[:, :, 2])
        oth_circles = torch.stack([oth_front, oth_rear], dim=2)
        diff = ego_circles[:, None, :, None, :] - oth_circles[:, :, None, :, :]
        dist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)
        pair_pen = F.relu(self.D_veh_safe - dist).pow(2)
        per_veh_max = pair_pen.flatten(2).max(dim=2).values
        veh_violation = (per_veh_max * validity).sum(dim=1)

        ref_start = self.DIM_EGO + self.DIM_OTHERS + self.DIM_VALIDITY
        lat = s[:, ref_start + 3]  # 横向误差 (DIM_REF_ERROR[3])
        temp_start = ref_start + self.DIM_REF_ERROR
        temporal_idx = s[:, temp_start:temp_start + self.DIM_TEMPORAL].squeeze(-1).long()
        road_dist_ref = torch.clamp(self.state_builder.get_road_dist_batch(
            w_i, temporal_idx, self.ego_indices, p_i if p_i is not None else [0] * len(w_i), s.device), max=50.0)
        ego_road_dist = road_dist_ref - torch.abs(lat)
        road_violation = F.relu(self.D_road_safe - ego_road_dist)

        if getattr(self.config, 'no_veh_penalty', False):
            veh_violation = torch.zeros_like(veh_violation)
        if getattr(self.config, 'no_road_penalty', False):
            road_violation = torch.zeros_like(road_violation)

        return veh_violation + road_violation

    def update(self):
        windows, word_indexs, path_indices = self.buffer.sample_batch(self.batch_size)
        if len(windows) == 0:
            return None, None
        states_tensor = torch.from_numpy(windows[:, 0, :]).to(self.device)  # window=1: 取第0帧

        # PEV
        critic_loss = self.update_critic(states_tensor, word_indexs, path_indices)
        self.pev_step += 1

        # PIM (经典 GEP 累加)
        actor_loss = None
        if self.pev_step >= self.pim_interval:
            self.pev_step = 0
            actor_loss, has_violation, max_p = self.update_actor(states_tensor, word_indexs, path_indices)
            if has_violation:
                self.rho = min(self.rho + self.amplifier_c, self.max_penalty)
                self.gep_iteration += 1
                logger.info(f'[GEP] violation, rho={self.rho:.4f} gep={self.gep_iteration}')

        return critic_loss, actor_loss

    def update_ego_indices(self, new_ego_indices):
        self.ego_indices = new_ego_indices

    def clear_buffer(self):
        self.buffer = PERBuffer(
            capacity=self.config.buffer_capacity,
            min_start_train=self.config.batch_size,
            window_size=1,
            state_dim=self.TOTAL_STATE_DIM,
        )

    def save(self, save_info: Dict[str, Any]) -> None:
        if self.globe_eps % self.config.save_freq == 0:
            logger.info(f'保存模型: globe_eps={self.globe_eps}, global_step={self.global_step}')
            model = {'actor': self.actor, 'critic': self.critic}
            optimizer = {'actor_optim': self.actor_optimizer, 'critic_optim': self.critic_optimizer}
            extra_info = {
                'config': self.config, 'global_step': self.global_step, 'history': self.history_loss,
                'globe_eps': self.globe_eps, 'state_dim': self.TOTAL_STATE_DIM,
                'rho': self.rho, 'gep_iteration': self.gep_iteration,
                'actor_scheduler_last_epoch': self.actor_scheduler.last_epoch,
                'critic_scheduler_last_epoch': self.critic_scheduler.last_epoch,
            }
            save_checkpoint(model=model, model_name='idc-baseline',
                            optimizer=optimizer, file_dir=self.config.file_dir,
                            env_name=save_info.get('env_name', 'unknown_env'),
                            extra_info=extra_info, metrics={'episode': extra_info['globe_eps']})

    def load(self, path: str) -> Dict[str, Any]:
        checkpoint = load_checkpoint(
            model={'actor': self.actor, 'critic': self.critic},
            filepath=path,
            optimizer={'actor_optim': self.actor_optimizer, 'critic_optim': self.critic_optimizer},
            device=self.device)
        self.globe_eps = checkpoint['globe_eps']
        self.history_loss = checkpoint['history']
        self.epoch_history_pdms = checkpoint.get('history_pdms', [])
        self.global_step = checkpoint['global_step']
        self.rho = checkpoint['rho'] == 0.0 and self.config.init_penalty or checkpoint['rho']
        self.gep_iteration = checkpoint['gep_iteration']
        for pg in self.actor_optimizer.param_groups:
            pg['lr'] = self.config.lr_actor_max
        for pg in self.critic_optimizer.param_groups:
            pg['lr'] = self.config.lr_critic_max
        last_ep_a = checkpoint.get('actor_scheduler_last_epoch', 0)
        last_ep_c = checkpoint.get('critic_scheduler_last_epoch', 0)
        self.actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.actor_optimizer, T_max=self.actor_scheduler.T_max,
            eta_min=self.actor_scheduler.eta_min, last_epoch=last_ep_a)
        self.critic_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.critic_optimizer, T_max=self.critic_scheduler.T_max,
            eta_min=self.critic_scheduler.eta_min, last_epoch=last_ep_c)
        return checkpoint
