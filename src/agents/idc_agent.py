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
        self.DIM_REF_ERROR = 3            # [delta_p, delta_phi, delta_v]
        self.TOTAL_STATE_DIM = self.DIM_EGO + self.DIM_OTHERS + self.DIM_REF_ERROR
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
        self.pos_err_weight = 0.2
        self.speed_err_weight = 0.01
        self.heading_err_weight = 0.3
        self.steer_cost_weight = 0.1
        self.acc_cost_weight = 0.005

        # GEP 惩罚
        self.rho = args.init_penalty
        self.max_penalty = args.max_penalty
        self.amplifier_c = args.amplifier_c
        self.gamma = 0.99
        self.pev_step = 0
        self.pim_interval = args.pim_interval

        # 安全距离
        self.D_veh_safe = 2.0    # 两车圆心威胁距离: 并排邻车道 ~3.75m 不受罚
        self.D_road_safe = 1.0
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
        self.noise_std = 0.2
        self.noise_decay_rate = 0.95
        self.noise_std_min = 0.05

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

            # 转向映射：[-1,1] → [-0.4,0.4] rad
            delta_phy = norm_action[..., 0] * 0.4

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
            l = self.utility_batch(s, u, w_i, p_i)
            if torch.isnan(l).any():
                logger.warning(f'[NaN] utility at rollout step {t}')
            l_min_log = min(l_min_log, l.min().item())
            l_max_log = max(l_max_log, l.max().item())
            s = self.f_pred_batch(s, u, w_i, p_i)
            if torch.isnan(s).any():
                logger.warning(f'[NaN] f_pred state at rollout step {t}')
            total_utility = total_utility + (self.gamma ** t) * torch.clamp(l, -10.0, 10.0)
        if self.global_step % 50 == 0:
            tu = total_utility
            logger.debug(f'[DIAG-critic] utility raw range=[{l_min_log:.2f}, {l_max_log:.2f}], '
                        f'target raw mean={tu.mean().item():.2f} min={tu.min().item():.2f} max={tu.max().item():.2f}')
        return torch.clamp(total_utility, -250.0, 250.0)

    def batch_state_to_tensor(self, states):
        logger.debug(f'将状态对象列表转换为张量表示: batch_size={len(states)},第一个状态={states[0] if len(states) > 0 else "N/A"}')
        # 一次性将所有状态转为张量
        states_tensors = [torch.from_numpy(s) for s in states]   # list of [TOTAL_STATE_DIM]
        
        ego_tensors = torch.stack([s[:self.DIM_EGO] for s in states_tensors]).to(self.device)  # [batch, DIM_EGO]
        others_tensors = torch.stack([s[self.DIM_EGO:self.DIM_EGO+self.DIM_OTHERS] for s in states_tensors]).to(self.device)
        ref_error_tensors = torch.stack([s[self.DIM_EGO+self.DIM_OTHERS:self.DIM_EGO+self.DIM_OTHERS+self.DIM_REF_ERROR] for s in states_tensors]).to(self.device)
        
        others_flat = others_tensors.view(len(states), -1)
        state_tensor = torch.cat([ego_tensors, others_flat, ref_error_tensors], dim=-1)
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
        
        # 1. 动力学推演
        ego_next = self.dynamics(ego_tensors[...,:6], actions)
        x_raw, y_raw, theta_next, v_next = ego_next[:, 0], ego_next[:, 1], ego_next[:, 2], ego_next[:, 3]

        # 2. 参考点查询 (必须先查 ref，才能以 ref 为锚点 clamp)
        refs = self.state_builder.get_ref_states_batch(
            w_i, x_raw.detach(), y_raw.detach(), self.ego_indices, p_i)
        ref_x, ref_y = refs[:, 0], refs[:, 1]

        # 3. 以参考点为锚点 clamp 位置：防止推演异常跳变，不影响正常跟踪
        max_tracking_dist = 50.0  # 25步推演正常位移 < 37m，50m 预留裕量
        x_next = torch.clamp(x_raw, ref_x - max_tracking_dist, ref_x + max_tracking_dist)
        y_next = torch.clamp(y_raw, ref_y - max_tracking_dist, ref_y + max_tracking_dist)

        # 诊断：clamp 实际触发（|raw - clamped| > 1m）
        clamp_x = (torch.abs(x_raw - x_next) > 1.0)
        clamp_y = (torch.abs(y_raw - y_next) > 1.0)
        if (clamp_x | clamp_y).any():
            idxs = (clamp_x | clamp_y).nonzero(as_tuple=True)[0].tolist()
            for i in idxs[:5]:
                logger.debug(f'[FPRED-CLAMP] sample_{i} world_{w_i[i]} raw=({x_raw[i].item():.1f},{y_raw[i].item():.1f}) '
                               f'ref=({ref_x[i].item():.1f},{ref_y[i].item():.1f}) '
                               f'-> clamped=({x_next[i].item():.1f},{y_next[i].item():.1f})')

        # 4. 组装 next_state
        ego_next_formatted = torch.stack([
            x_next, y_next, 
            v_next, torch.zeros_like(v_next),  # v_lon, v_lat
            theta_next, torch.zeros_like(theta_next) # phi, omega
        ], dim=-1)
        
        others_next = self.predict_others(others_tensors)
        
        # 5. 参考误差（用 clamped 坐标）
        dx = ref_x - x_next
        dy = ref_y - y_next
        delta_p = torch.hypot(dx, dy)
        cross = dy * torch.cos(refs[:, 4]) - dx * torch.sin(refs[:, 4])
        sign = torch.where(cross >= 0, torch.tensor(1.0, device=self.device),
                           torch.tensor(-1.0, device=self.device))
        delta_p = delta_p * sign

        # 诊断：clamp 后仍大偏离
        if (delta_p.abs() > 100.0).any():
            big = (delta_p.abs() > 100.0).nonzero(as_tuple=True)[0].tolist()
            for i in big[:3]:
                logger.warning(f'[FPRED-ERR] sample_{i} delta_p={delta_p[i].item():.1f}m '
                               f'ego=({x_next[i].item():.1f},{y_next[i].item():.1f}) '
                               f'ref=({ref_x[i].item():.1f},{ref_y[i].item():.1f})')

        delta_phi = theta_next - refs[:, 4]
        delta_phi = torch.atan2(torch.sin(delta_phi), torch.cos(delta_phi))
        
        ego_speed = torch.hypot(ego_next_formatted[:, 2], ego_next_formatted[:, 3])
        delta_v = ego_speed - refs[:, 2]
        
        ref_error_tensors = torch.stack([delta_p, delta_phi, delta_v], dim=-1)
        
        # 组合并返回最终状态
        next_states = []
        for i in range(len(states)):
            next_s = self.tensor_to_state_tensor(ego_next_formatted[i], others_next[i], ref_error_tensors[i])
            next_states.append(next_s)
            
        return torch.stack(next_states)
    
    def tensor_to_state_tensor(self, ego_tensor, others_tensor, ref_error_tensor):
        """
        将 ego_tensor 和 others_tensor ref_error_tensor 转成网络能接收的tensor格式的状态对象
        ego_next shape=torch.Size([10, 4]), others_next shape=torch.Size([10, 8, 4]), ref_error shape=torch.Size([10, 3])
        """
        return torch.cat([ego_tensor, others_tensor.view(-1), ref_error_tensor], dim=-1)  # [TOTAL_STATE_DIM]


    def update_critic(self, states, world_indices, path_indices=None):
        logger.debug(f'更新 Critic: states batch size={states.shape[0]}')
        with torch.no_grad():
            targets = self.compute_rollout_target(states, world_indices, path_indices).detach()
        logger.info(f"[DEBUG] target stats: min={targets.min().item():.2f}, max={targets.max().item():.2f}, mean={targets.mean().item():.2f}, has_nan={torch.isnan(targets).any()}")

        if torch.isnan(targets).any():
            logger.warning(f'Target 包含 NaN，跳过本次 Critic 更新')
            return float('nan')

        values = self.critic(states)
        logger.info(f"[DEBUG] value stats: min={values.min().item():.2f}, max={values.max().item():.2f}, mean={values.mean().item():.2f}, has_nan={torch.isnan(values).any()}")

        loss = F.mse_loss(values, targets.unsqueeze(1))
        logger.info(f"[DEBUG] loss = {loss.item():.4f}")

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
        for t in range(self.horizon):
            u = self.actor(s)
            l = self.utility_batch(s, u, w_i, p_i)
            p = self.penalty_batch(s, w_i)
            if t == 0:
                logger.debug(f'[DIAG-pen] raw penalty min={p.min().item():.2f} max={p.max().item():.2f} '
                            f'mean={p.mean().item():.2f}  (rho={self.rho:.4f})')
            l = torch.clamp(l, -10.0, 10.0)
            total_cost = total_cost + (self.gamma ** t) * (l + self.rho * p)
            s = self.f_pred_batch(s, u, w_i, p_i)

        total_cost = torch.clamp(total_cost, -100.0, 100.0)

        actor_loss = total_cost.mean()

        # 衰减探索噪声
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
        p_i: 可选，list of int，每条数据的候选路径索引
        返回 [batch] 标量代价
        """
        ego = s[:, :self.DIM_EGO]
        refs = self.state_builder.get_ref_states_batch(
            w_i, ego[:, 0], ego[:, 1], self.ego_indices, p_i)

        pos_err = torch.sqrt((ego[:, 0] - refs[:, 0]) ** 2
                            + (ego[:, 1] - refs[:, 1]) ** 2 + 1e-8)
        heading_err = ego[:, 4] - refs[:, 4]
        heading_err = torch.atan2(torch.sin(heading_err), torch.cos(heading_err))
        speed_err = torch.sqrt((ego[:, 2] - refs[:, 2]) ** 2
                              + (ego[:, 3] - refs[:, 3]) ** 2 + 1e-8)

        steer_cost = u[:, 0] ** 2
        acc_cost = u[:, 1] ** 2

        l = (self.pos_err_weight * pos_err ** 2
             + self.speed_err_weight * speed_err ** 2
             + self.heading_err_weight * heading_err ** 2
             + self.steer_cost_weight * steer_cost
             + self.acc_cost_weight * acc_cost)
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
        if self.global_step % 50 == 0:
            logger.debug(f'[DIAG-pos] ego=({ego[0,0].item():.2f},{ego[0,1].item():.2f}) '
                        f'oth0=({others[0,0,0].item():.2f},{others[0,0,1].item():.2f}) '
                        f'min_dist={dist[0,:,:,:].min().item():.4f} max_dist={dist[0,:,:,:].max().item():.4f}')
        # 每辆周车取和最危险的一对圆碰撞，再跨所有周车求和
        pair_pen = F.relu(self.D_veh_safe - dist).pow(2)          # [batch, 8, 2, 2]
        veh_violation = pair_pen.flatten(2).max(dim=2).values.sum(dim=1)  # [batch]

        # 道路边界惩罚
        edge_pts = self.state_builder.get_road_edges_batch(
            w_i, ego[:, 0], ego[:, 1])
        edge_dist = torch.sqrt((edge_pts[:, 0] - ego[:, 0]) ** 2
                              + (edge_pts[:, 1] - ego[:, 1]) ** 2 + 1e-8)
        road_violation = F.relu(self.D_road_safe - edge_dist).pow(2)

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
        

    def save(self, save_info: Dict[str, Any]) -> None:
        # 满足回合数才会保存模型
        if self.globe_eps % self.args.save_freq == 0:
            logger.info(f'保存模型: globe_eps={self.globe_eps}, global_step={self.global_step}')
            model = {'actor': self.actor, 'critic': self.critic}
            optimizer = {'actor_optim': self.actor_optimizer, 'critic_optim': self.critic_optimizer}
            # 先加载历史损失数据，如果有的话
            self.history_loss.append(save_info.get('history_loss', []).copy())
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
        self.rho = checkpoint['rho']
        self.gep_iteration = checkpoint['gep_iteration']
        return checkpoint
        