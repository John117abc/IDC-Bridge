import sys
from turtle import distance
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
    def __init__(self, env, args, device):
        self.env = env
        self.device = device
        self.num_worlds = env.num_worlds
        self.max_agents = env.max_cont_agents

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
        self.buffer = PERBuffer(capacity=100000, min_start_train=100)
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
            logger.info(f'选择动作: state_tensor shape={state_tensor.shape}')
            
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


    def predict_others(others_state, dt):
        """
        others_state: [batch, N, 4]  (px, py, phi, vlon)
        返回下一时刻的 others_state
        """
        px, py, phi, vlon = others_state.unbind(dim=-1)
        px_next = px + vlon * torch.cos(phi) * dt
        py_next = py + vlon * torch.sin(phi) * dt
        phi_next = phi
        vlon_next = vlon
        return torch.stack([px_next, py_next, phi_next, vlon_next], dim=-1)
    
    def compute_rollout_target(self, states):
        """
        states: list of state objects  (batch)
        返回长度为 batch 的 tensor，每个元素是累计效用
        """
        batch = len(states)
        # 假设路径信息可以用连续表示（如跟踪误差已经在 utility_fn 里动态计算）
        # 为简化，我们直接在 rollout 循环中调用 utility_fn，它需要原始状态对象
        # 因此更好的做法是保持状态为对象列表，但在计算图中走不通。
        # 实际实现时，我们会将路径信息也编码为张量（如相对路径点）。
        # 这里给出概念性伪代码：
        total_utility = torch.zeros(batch)
        s = states   # 原始对象列表
        for t in range(self.horizon):
            # 从当前状态计算动作
            u = self.actor( self.batch_state_to_tensor(s) )   # [batch, 2]
            # 计算即时效用（需要针对每个样本调用 utility）
            l = torch.tensor([self.utility(s_i, u_i) for s_i, u_i in zip(s, u)])
            total_utility += l
            # 使用预测模型 f_pred 更新状态（这里需要更新对象中的数值）
            s = self.f_pred_batch(s, u)   # 返回新的状态对象列表（或张量）
        return total_utility

    def batch_state_to_tensor(self, states):
        logger.info(f'将状态对象列表转换为张量表示: batch_size={len(states)}')
        # 一次性将所有状态转为张量
        states_tensors = [torch.from_numpy(s) for s in states]   # list of [TOTAL_STATE_DIM]
        
        ego_tensors = torch.stack([s[:self.DIM_EGO] for s in states_tensors]).to(self.device)  # [batch, DIM_EGO]
        others_tensors = torch.stack([s[self.DIM_EGO:self.DIM_EGO+self.DIM_OTHERS] for s in states_tensors]).to(self.device)
        ref_error_tensors = torch.stack([s[self.DIM_EGO+self.DIM_OTHERS:self.DIM_EGO+self.DIM_OTHERS+self.DIM_REF_ERROR] for s in states_tensors]).to(self.device)
        
        others_flat = others_tensors.view(len(states), -1)
        state_tensor = torch.cat([ego_tensors, others_flat, ref_error_tensors], dim=-1)
        return state_tensor
    
    def f_pred_batch(self,states, actions):
        """
        states: list of state objects (batch)
        actions: tensor [batch, 2]
        返回新的状态对象列表（或张量）
        """
        # 将 states 转为张量表示
        ego_tensors = torch.stack([s.ego_tensor for s in states])        # [batch, 6]
        others_tensors = torch.stack([s.others_tensor for s in states])  # [batch, N, 4]

        # 使用自行车模型预测下一时刻自车状态
        ego_next = self.dynamics(ego_tensors[...,:6], actions)  # [batch, 6]

        # 使用简单的运动学模型预测周车状态（假设动作对周车无影响）
        others_next = self.predict_others(others_tensors, self.dt)  # [batch, N, 4]

        # 将 ego_next 和 others_next 转回状态对象列表（需要设计 tensor_to_state 函数）
        next_states = []
        for i in range(len(states)):
            next_s = self.tensor_to_state(ego_next[i], others_next[i], states[i].path_ref)
            next_states.append(next_s)
        return next_states
    
    def tensor_to_state(self, ego_tensor, others_tensor, path_ref):
        """
        将 ego_tensor 和 others_tensor 转回状态对象。
        需要根据你的状态对象定义来实现。
        """
        network_state = torch.cat([ego_tensor, others_tensor.view(-1), path_ref], dim=-1)  # [TOTAL_STATE_DIM]
        return network_state

    def update_critic(self,states):
        targets = self.compute_rollout_target(states)
        values = self.critic(self.batch_state_to_tensor(states))
        loss = F.mse_loss(values, targets)
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

    def update_actor(self,states):
        # 使用当前 actor 和模型展开 H 步，得到累计效用 + 惩罚项
        total_cost = 0.0
        s = states
        for t in range(self.horizon):
            u = self.actor(self.batch_state_to_tensor(s)).sample()   # 重参数化动作
            # 效用
            l = torch.tensor([self.utility(s_i, u_i) for s_i, u_i in zip(s, u)])
            # 惩罚项
            p = torch.tensor([self.penalty(s_i) for s_i in s])
            total_cost += (l + self.rho * p)
            s = self.f_pred_batch(s, u)
        # 目标是最小化 total_cost，因此 actor 的 loss = total_cost.mean()
        # 但为了利用 critic 降低方差，也可以使用 advantage，这里直接最小化 cost 本身
        actor_loss = total_cost.mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def utility(self, s, u):
        """
        s: 状态对象（包含 ego, others, ref）
        u: [delta, a] 控制量
        返回标量代价 l = 跟踪误差 + 控制能量
        """
        # 从 s 中获取自车状态和参考路径
        ego = s.ego
        ref = s.path_ref
        # 计算跟踪误差
        pos_err = distance(ego[0:2], ref.closest_point(ego[0:2]))
        heading_err = ego[4] - ref.closest_heading(ego[0:2])
        speed_err = ego[2] - ref.desired_speed(ego[0:2])
        # 控制能量
        steer_cost = u[0]**2
        acc_cost = u[1]**2
        # 加权和（论文中系数）
        l = (self.pos_err_weight * pos_err**2 +
            self.speed_err_weight * speed_err**2 +
            self.heading_err_weight * heading_err**2 +
            self.steer_cost_weight * steer_cost +
            self.acc_cost_weight * acc_cost)
        return l

    def penalty(self, s):
        """
        计算状态 s 下的约束违反惩罚（用于 Actor 更新）
        包括：与周围车辆安全距离、与道路边界安全距离、红灯停止线
        返回标量惩罚值（非负）
        """
        violation = 0.0
        # ego = s.ego
        # # 与其他车辆的碰撞约束
        # for other in s.others:
        #     dist = euclidean_distance(ego[0:2], other[0:2])
        #     safe_dist = self.D_veh_safe
        #     if dist < safe_dist:
        #         violation += (safe_dist - dist)**2
        # # 与道路边界约束
        # road_boundary_dist = distance_to_road_boundary(ego[0:2], s.path_ref)
        # if road_boundary_dist < self.D_road_safe:
        #     violation += (self.D_road_safe - road_boundary_dist)**2
        # 红灯停止线（简化：如果红灯且超越停止线）
        # if s.traffic_light == 'red' and ego[0] > stop_line_position(s.path_ref):
        #     violation += (ego[0] - stop_line_position)**2
        return violation
    
    def update(self):
        states = self.buffer.sample_batch(self.batch_size)
        # 1. 更新 Critic
        self.update_critic(states)
        # 2. 更新 Actor（每隔几次 Critic 更新）
        if self.global_step % self.args.update_freq == 0:
            self.rho = min(self.init_penalty * (self.amplifier_c ** (self.gep_iteration // self.amplifier_m)), self.max_penalty)
            self.update_actor(states)
            self.gep_iteration += 1