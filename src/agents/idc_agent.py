import sys
import numpy as np
import torch
import torch.optim as optim

# 导入 GPUDrive 相关
from gpudrive.env.env_torch import GPUDriveTorchEnv
from typing import Tuple

# 导入 IDC 的核心模块（你需要保证这些文件在 path 里）
from models.actor_critic import DiscreteActor, DiscreteCritic
from models.bicycle import BicycleModel
from buffer import PERBuffer
from utils import DiscreteActionMapper


# ==============================================
# IDC Agent 适配类（离散动作版本）
# ==============================================
class DiscreteIDCAgent:
    """
    基于离散动作的 IDC 智能体，保留原始 IDC 的网络结构与 GEP 训练逻辑，
    但动作输出改为离散 logits，并在与环境交互时通过映射器转成物理量。
    """
    def __init__(self, env: GPUDriveTorchEnv, args, device):
        self.env = env
        self.device = device
        self.num_worlds = env.num_worlds
        self.max_agents = env.max_cont_agents

        # 读取 IDC 超参数（沿用你原来的配置文件）
        self.DIM_EGO = 6
        self.DIM_OTHER = 32   # 8 车 × 4
        self.DIM_REF_ERROR = 3
        self.TOTAL_STATE_DIM = self.DIM_EGO + self.DIM_OTHER + self.DIM_REF_ERROR
        self.DIM_ROAD = 80

        # 超参
        self.dt = args.dt
        self.horizon = args.horizon
        self.batch_size = args.batch_size

        # 网络
        self.actor = DiscreteActor(self.TOTAL_STATE_DIM, args.hidden_dim).to(device)
        self.critic = DiscreteCritic(self.TOTAL_STATE_DIM, args.hidden_dim).to(device)
        self.dynamics = BicycleModel(dt=self.dt, L=2.9)

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.lr_critic)

        # 动作映射器
        self.action_mapper = DiscreteActionMapper(steer_bins=13, accel_bins=7)

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

        # 其他
        self.gamma = 0.99
        self.buffer = PERBuffer(capacity=100000, min_start_train=256)
        self.global_step = 0
        self.gep_iteration = 0

        # 道路状态缓存（每 world 单独存）
        self.road_states = [None] * self.num_worlds

    def select_action(self, state: torch.Tensor, deterministic: bool = False):
        """
        输入观测 state [batch, TOTAL_STATE_DIM]，返回物理动作 [batch, 2] 和一维动作索引 [batch]
        """
        state = state.to(self.device)
        with torch.no_grad():
            logits = self.actor(state)  # 输出 (steer_logits, accel_logits) 或自定义
        # (steer_logits, accel_logits)
        steer_logits, accel_logits = logits
        if deterministic:
            steer_idx = torch.argmax(steer_logits, dim=1)
            accel_idx = torch.argmax(accel_logits, dim=1)
        else:
            steer_dist = torch.distributions.Categorical(logits=steer_logits)
            accel_dist = torch.distributions.Categorical(logits=accel_logits)
            steer_idx = steer_dist.sample()
            accel_idx = accel_dist.sample()

        # 转为物理量
        actions_phy = []
        for b in range(state.shape[0]):
            act = self.action_mapper.index_to_action(steer_idx[b].item(), accel_idx[b].item())
            actions_phy.append(act)
        actions_phy = np.stack(actions_phy)

        # 一维动作索引（用于 env.step_dynamics）
        action_1d = steer_idx * self.action_mapper.accel_bins + accel_idx
        return actions_phy, action_1d.cpu().numpy()

    def update(self, ref_paths: torch.Tensor, road_states_batch: torch.Tensor):
        """
        IDC GEP 更新，逻辑与原始 idc_agent.py 类似，但输入改成批量的 ref_path 和 road_state。
        注意：这里需要你结合原来的 _forward_horizon 实现。
        """
        if len(self.buffer) < self.batch_size:
            return

        batch = self.buffer.sample_batch(self.batch_size)
        states, _, _, _, _, infos = zip(*batch)
        state_tensor = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)

        # 为每个样本取对应的 ref_path 和 road_state
        # 这里简化写：假设所有样本共享同一个 ref_path？对于多 world 批次，需要从 info 里提取。
        # 实际使用时，你需要从 info 中拿 per-sample 的 ref_path 和 road_state。
        # 为了示例，暂时用传入的 ref_paths 和 road_states_batch（batch 内维度需对齐）。
        # 建议在收集经验时将 ref_path 和 road_state 存储到 info 中，这里从 batch 表取出。

        # --- Critic update ---
        # (复用原 IDC 的 _forward_horizon 逻辑，这里仅示意)
        # ...

        # --- Actor update ---
        # ...

        # --- GEP penalty amplify ---
        # ...

        return None  # 后续可返回 loss 等

    def collect_experience(self, builder, ego_indices, obs_batch, actions_phy, rewards, dones, infos):
        """
        从所有 world 收集经验到 buffer。
        obs_batch: [num_worlds, max_agents, obs_dim] 但只取 ego 的观测
        我们需要把每个 world 的 ego 观测、动作、奖励等拆开存入 buffer。
        """
        for w in range(self.num_worlds):
            ego_idx = ego_indices[w]
            obs = obs_batch[w, ego_idx].cpu().numpy()
            next_obs = None  # 需要 step 后获取，此处仅为示意
            action = actions_phy[w]
            reward = rewards[w]
            done = dones[w]
            info = infos[w]
            # 存入 buffer（格式要与原 IDC 一致）
            self.buffer.push(obs, action, reward, next_obs, done, info)