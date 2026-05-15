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

logger = get_logger('idc-agent')


# ==============================================
# IDC Agent 适配类（离散动作 + Gumbel-Softmax）
# ==============================================
class DiscreteIDCAgent:
    def __init__(self, env, args, device,state_builder,ego_indices):
        self.env = env
        self.device = device
        self.num_worlds = env.num_worlds
        self.max_agents = env.max_cont_agents
        self.state_builder = state_builder  # 传入状态构建器实例
        self.ego_indices = ego_indices  # 每个世界中自车的索引列表

        # IDC 维度定义
        self.DIM_EGO = 6                  # [x, y, v_lon, v_lat, phi, omega] (omega=0)
        self.DIM_OTHERS = 32               # 8 车 × 4 (x, y, phi, v_lon)
        self.DIM_REF_ERROR = 3            # [delta_p, delta_phi, delta_v]
        self.TOTAL_STATE_DIM = self.DIM_EGO + self.DIM_OTHERS + self.DIM_REF_ERROR
        self.DIM_ROAD = 80

        # 超参
        self.dt = args.dt
        self.horizon = args.horizon
        self.batch_size = args.batch_size
        self.update_freq = args.update_freq

        # 网络
        self.actor = ContinuousActor(self.TOTAL_STATE_DIM, args.hidden_dim).to(device)
        self.critic = ContinuousCritic(self.TOTAL_STATE_DIM, args.hidden_dim).to(device)
        self.dynamics = KinematicBicycleModel(dt=self.dt)

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.lr_critic)

        # IDC 成本权重
        self.pos_err_weight = 0.04
        self.speed_err_weight = 0.01
        self.heading_err_weight = 0.1
        self.steer_cost_weight = 0.1
        self.acc_cost_weight = 0.005

        # GEP 惩罚
        self.rho = args.init_penalty
        self.max_penalty = args.max_penalty
        self.amplifier_c = args.amplifier_c
        self.amplifier_m = args.amplifier_m
        self.gamma = 0.99

        # 安全距离
        self.D_veh_safe = 0.5   # 米，可根据需要调整
        self.D_road_safe = 0.3
        self.HALF_L = 2.25
        self.HALF_W = 1.0

        # 缓冲区
        self.buffer = PERBuffer(capacity=100000, min_start_train=args.batch_size)
        self.global_step = 1
        self.gep_iteration = 0

        # 参考速度
        self.ref_vlon = 5.0

        # 道路状态缓存，键为 world_idx
        self.road_states = [None] * self.num_worlds


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
                noise = torch.normal(0, 0.1, size=norm_action.shape, device=norm_action.device)
                norm_action = norm_action + noise
                norm_action = torch.clamp(norm_action, -1.0, 1.0)

            # 转向映射：[-1,1] → [-0.4,0.4] rad
            delta_phy = norm_action[..., 0] * 0.4

            # 加速度映射：线性插值 [-1,1] → [-3.0, 1.5]
            a_phy = (norm_action[..., 1] + 1) / 2 * (1.5 - (-3.0)) + (-3.0)   # 范围 [-3.0, 1.5]

            # 组合最终动作 [batch, max_agents, 2]
            action = torch.stack([delta_phy, a_phy], dim=-1)    # 或 torch.cat 后 reshape
            return action


    def predict_others(self,others_state):
        """
        others_state: [batch, N * 4]  (px, py, phi, vlon)
        返回下一时刻的 others_state
        """
        logger.debug(f'预测周车下一状态: others_state shape={others_state.shape}')
        px, py, phi, vlon = others_state.view(others_state.shape[0], int(self.DIM_OTHERS / 4), 4).unbind(dim=-1)  # [batch, N]
        px_next = px + vlon * torch.cos(phi) * self.dt
        py_next = py + vlon * torch.sin(phi) * self.dt
        phi_next = phi
        vlon_next = vlon
        return torch.stack([px_next, py_next, phi_next, vlon_next], dim=-1)
    
    def compute_rollout_target(self, states,word_indexs,step_counts):
        """
        states: list of state objects  (batch)
        返回长度为 batch 的 tensor，每个元素是累计效用
        """
        logger.debug(f'计算 rollout 目标: states batch size={len(states)}')
        batch = len(states)
        # 假设路径信息可以用连续表示（如跟踪误差已经在 utility_fn 里动态计算）
        # 为简化，我们直接在 rollout 循环中调用 utility_fn，它需要原始状态对象
        total_utility = torch.zeros(batch).to(self.device)
        s = states   # 原始对象列表
        w_i = word_indexs
        s_c = list(step_counts)
        for t in range(self.horizon):
            # 从当前状态计算动作
            logger.debug(f' s 的形状: {s.shape if isinstance(s, torch.Tensor) else "list of length " + str(len(s))}')
            u = self.actor(s)   # [batch, 2]
            # 计算即时效用并且对应的s_c需要加一
            l_list = []
            for i, (s_i, u_i) in enumerate(zip(s, u)):
                val = self.utility(s_i, u_i, w_i[i], s_c[i])
                l_list.append(val)
                s_c[i] += 1        
            l = torch.stack(l_list)          # [batch]，每个元素保持梯度            total_utility += l
            # 使用预测模型 f_pred 更新状态（这里需要更新对象中的数值）
            s = self.f_pred_batch(s, u)   # 返回新的状态对象列表（或张量）
        return total_utility

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
    
    def f_pred_batch(self, states, actions):
        """
        states: list of state objects (batch)
        actions: tensor [batch, 2]
        返回新的状态对象列表（或张量）
        """
        # 将 states 转为张量表示
        ego_tensors = states[:, :self.DIM_EGO]  # [batch, DIM_EGO]
        others_tensors = states[:, self.DIM_EGO:self.DIM_EGO+self.DIM_OTHERS]  # [batch, DIM_OTHERS]
        ref_error_tensors = states[:, self.DIM_EGO+self.DIM_OTHERS:self.DIM_EGO+self.DIM_OTHERS+self.DIM_REF_ERROR]  # [batch, DIM_REF_ERROR]
        logger.debug(f'预测下一状态: others_tensors shape={others_tensors.shape}')
        
        # 使用自行车模型预测下一时刻自车状态x_next, y_next, theta_next, v_next
        ego_next = self.dynamics(ego_tensors[...,:6], actions)
        
        # 自车预测状态转成[x, y, v_lon, v_lat, phi, omega]格式（假设omega=0）
        x_next = ego_next[:, 0]
        y_next = ego_next[:, 1]
        theta_next = ego_next[:, 2]
        v_next = ego_next[:, 3]
        v_lon_next = v_next * torch.cos(theta_next)
        v_lat_next = v_next * torch.sin(theta_next)
        ego_next_formatted = torch.stack([x_next, y_next, v_lon_next, v_lat_next, theta_next, torch.zeros_like(theta_next)], dim=-1)  # [batch, 6]

        # 使用简单的运动学模型预测周车状态（假设动作对周车无影响）
        others_next = self.predict_others(others_tensors)  # [batch, N, 4]
        # 将 ego_next 和 others_next 转回状态对象列表
        logger.debug(f'预测完成，构建下一状态对象列表: ego_next shape={ego_next.shape}, others_next shape={others_next.shape}, ref_error shape={ref_error_tensors.shape}')
        next_states = []
        for i in range(len(states)):
            next_s = self.tensor_to_state_tensor(ego_next_formatted[i], others_next[i], ref_error_tensors[i])
            next_states.append(next_s)
        return torch.stack(next_states)  # [batch, TOTAL_STATE_DIM]
    
    def tensor_to_state_tensor(self, ego_tensor, others_tensor, ref_error_tensor):
        """
        将 ego_tensor 和 others_tensor ref_error_tensor 转成网络能接收的tensor格式的状态对象
        ego_next shape=torch.Size([10, 4]), others_next shape=torch.Size([10, 8, 4]), ref_error shape=torch.Size([10, 3])
        """
        return torch.cat([ego_tensor, others_tensor.view(-1), ref_error_tensor], dim=-1)  # [TOTAL_STATE_DIM]


    def update_critic(self,states,word_indexs,step_counts):
        logger.debug(f'更新 Critic: states batch size={len(states)}')
        targets = self.compute_rollout_target(states,word_indexs,step_counts)
        # 截断目标值，防止过大导致训练不稳定
        targets = torch.clamp(targets, -100.0, 100.0)
        values = self.critic(states)
        loss = F.mse_loss(values, targets.unsqueeze(1))
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()
        return loss.detach().item()

    def update_actor(self, states, w_i, step_counts):
        logger.debug(f'更新 Actor: states batch size={len(states)}')
        total_cost = 0.0
        s = states
        s_c = list(step_counts)
        for t in range(self.horizon):
            u = self.actor(s)  # 假设 u 形状 [batch, action_dim]
            # 关键修改：使用 torch.stack 代替 torch.tensor，保留梯度
            l_list = []
            for i, (s_i, u_i) in enumerate(zip(s, u)):
                val = self.utility(s_i, u_i, w_i[i], s_c[i])
                l_list.append(val)
                s_c[i] += 1            
            l = torch.stack(l_list)          # [batch]，每个元素保持梯度
            p_list = [self.penalty(s_i, w_i[i]) for i, s_i in enumerate(s)]
            p = torch.stack(p_list)          # [batch]
            # 截断成本和惩罚值，防止过大导致训练不稳定
            l = torch.clamp(l, -10.0, 10.0)
            p = torch.clamp(p, 0.0, 10.0)
            # 打印控制消耗和惩罚值的统计信息
            # logger.info(f'Actor 更新: step {t}, 平均效用 {l.mean().item():.4f}, 平均惩罚 {p.mean().item():.4f}')
            total_cost += (l + self.rho * p)
            s = self.f_pred_batch(s, u)
        
        actor_loss = total_cost.mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return actor_loss.detach().item()

    def utility(self, s, u,w_i,step_count):
        """
        s: 状态对象（包含 ego, others, ref）
        u: [delta, a] 控制量
        w_i: 世界索引
        step_count: 步数计数器
        返回标量代价 l = 跟踪误差 + 控制能量
        """
        # 从 s 中获取自车状态和参考路径
        logger.debug(f'计算效用: 状态={s.shape}, 动作={u.shape}')
        ego = s[:self.DIM_EGO]  # [6]
        # 通过state_builder的w_i所以计算参考
        ref = self.state_builder.get_ref_state(w_i, self.ego_indices[w_i], step_count) 
        # 计算跟踪误差
        pos_err = torch.sqrt((ego[0] - ref[0])**2 + (ego[1] - ref[1])**2)  # 位置误差
        heading_err = ego[4] - ref[4]  # 航向误差
        speed_err = torch.sqrt((ego[2] - ref[2])**2 + (ego[3] - ref[3])**2)  # 速度误差
        # 控制能量
        steer_cost = u[0]**2
        acc_cost = u[1]**2
        # 加权和
        l = (self.pos_err_weight * pos_err**2 +
            self.speed_err_weight * speed_err**2 +
            self.heading_err_weight * heading_err**2 +
            self.steer_cost_weight * steer_cost +
            self.acc_cost_weight * acc_cost)
        # 如果损失过大打印一下数据
        # logger.info(f'参考信息：ref_x={ref[0]:.4f}, ref_y={ref[1]:.4f}, ref_vlon={ref[2]:.4f}, ref_phi={ref[4]:.4f}')
        # logger.info(f'位置误差: pos_err={pos_err:.4f}, heading_err={heading_err:.4f}, speed_err={speed_err:.4f}')
        # logger.info(f'效用计算: pos_err={pos_err:.4f}, heading_err={heading_err:.4f}, speed_err={speed_err:.4f}')
        # if l.item() > 10.0:
        #     logger.info(f'位置误差: pos_err={pos_err:.4f}, heading_err={heading_err:.4f}, speed_err={speed_err:.4f}')
        #     logger.info(f'效用计算: pos_err={pos_err:.4f}, heading_err={heading_err:.4f}, speed_err={speed_err:.4f}')
        return l

    def penalty(self, s, w_i):
        """
        计算状态 s 下的约束违反惩罚（用于 Actor 更新）
        包括：与周围车辆安全距离、与道路边界安全距离、红灯停止线
        返回标量惩罚值（非负）
        """
        violation = torch.tensor(0.0).to(self.device)
        ego = s[:self.DIM_EGO]  # [6]
        # 与其他车辆的碰撞约束
        for other in s[self.DIM_EGO:self.DIM_EGO+self.DIM_OTHERS].view(-1, 4):  # 每4维一个车辆状态
            dist = torch.sqrt((ego[0] - other[0])**2 + (ego[1] - other[1])**2)
            safe_dist = self.D_veh_safe
            if dist < safe_dist:
                logger.debug(f' Penalty: 车辆间距过近，dist={dist:.4f}, safe_dist={safe_dist:.4f}')
                violation += (safe_dist - dist)**2
        # 与道路边界约束
        closest_edge_point = self.state_builder.get_road_edges(w_i, ego_x=ego[0].detach().item(), ego_y=ego[1].detach().item())  # 假设返回与最近道路边界的距离
        # 计算自车与道路边界的距离
        road_boundary_dist = torch.sqrt((closest_edge_point[0] - ego[0])**2 + (closest_edge_point[1] - ego[1])**2)
        if road_boundary_dist < self.D_road_safe:
            violation += (self.D_road_safe - road_boundary_dist)**2
        # 红灯停止线，后续开发
        # if s.traffic_light == 'red' and ego[0] > stop_line_position(s.path_ref):
        #     violation += (ego[0] - stop_line_position)**2
        return violation
    
    def update(self):
        states,step_counts, word_indexs = zip(*self.buffer.sample_batch(self.batch_size))
        # 1. 更新 Critic
        # 首先处理把state转成tensor，然后计算 rollout 目标，最后更新 critic 网络
        states_tensor = self.batch_state_to_tensor(states)
        critic_loss = self.update_critic(states_tensor,word_indexs,step_counts)
        # 2. 更新 Actor（每隔几次 Critic 更新）
        actor_loss = None
        if self.global_step % self.update_freq == 0 and self.rho <= self.max_penalty:
            # 由于states_tensor已经经过了上面critic的计算图,所以这里
            logger.debug(f'更新 Actor: global_step={self.global_step}, states_tensor shape={states_tensor.shape}')
            self.rho = min(self.rho * self.amplifier_c, self.max_penalty)
            actor_loss = self.update_actor(states_tensor.detach(),word_indexs,step_counts)
            self.gep_iteration += 1
        
        return critic_loss, actor_loss
        