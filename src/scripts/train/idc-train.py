import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import imageio
import json

# 导入 GPUDrive 相关
from gpudrive.env.config import EnvConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.visualize.utils import img_from_fig
from typing import Tuple

# 导入已完成的观测构建器
sys.path.insert(0, '/workspace/idc/src')
from env.idc_state_builder import GPUDriveObservationBuilder

# 导入 IDC 的核心模块（你需要保证这些文件在 path 里）
from models.actor_critic import DiscreteActor, DiscreteCritic
from models.bicycle import BicycleModel
from buffer import PERBuffer


# ==============================================
# 离散动作映射工具
# ==============================================
class DiscreteActionMapper:
    """
    将 GPUDrive 的离散动作索引 (0 ~ 90) 映射到连续物理量 (accel, steer)，
    并支持反向查表（用于前向预测时从连续量找到最近的离散动作）。
    """
    def __init__(self, steer_bins=13, accel_bins=7,
                 steer_range=(-0.4, 0.4), accel_range=(-3.0, 1.5)):
        self.steer_bins = steer_bins
        self.accel_bins = accel_bins
        self.steer_edges = torch.linspace(steer_range[0], steer_range[1], steer_bins)
        self.accel_edges = torch.linspace(accel_range[0], accel_range[1], accel_bins)

    def index_to_action(self, steer_idx: int, accel_idx: int) -> np.ndarray:
        """将离散索引转为物理动作 [accel, steer]"""
        accel = self.accel_edges[accel_idx].item()
        steer = self.steer_edges[steer_idx].item()
        return np.array([accel, steer], dtype=np.float32)

    def action_to_index(self, accel: float, steer: float) -> Tuple[int, int]:
        """将物理动作映射回最近的离散索引"""
        steer_idx = torch.argmin(torch.abs(self.steer_edges - steer)).item()
        accel_idx = torch.argmin(torch.abs(self.accel_edges - accel)).item()
        return steer_idx, accel_idx

    def full_action_idx_to_single(self, idx: int) -> Tuple[int, int]:
        """
        GPUDrive 通常把 (steer, accel) 展平为一维索引：
        总动作数 = steer_bins * accel_bins
        返回 (steer_idx, accel_idx)
        """
        steer_idx = idx // self.accel_bins
        accel_idx = idx % self.accel_bins
        return steer_idx, accel_idx


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
        # 需要根据你的 ActorNet 输出调整；假设它输出 (steer_logits, accel_logits)
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


# ==============================================
# 训练主循环
# ==============================================
def train(args):
    import os
    print("Data root:", args.data_dir)
    print("Files found:", os.listdir(args.data_dir))
    # 1. 数据加载器
    data_loader = SceneDataLoader(
        root=args.data_dir,
        batch_size=args.num_worlds,
        dataset_size=args.num_worlds,  # 或者更大，比如 100
        sample_with_replacement=True,  # 改成 True
        shuffle=True,
        seed=args.seed
    )

    # 2. 环境配置（与训练模型时保持一致）
    env_config = EnvConfig(
        ego_state=True,
        road_map_obs=False,      # 我们用自己的 builder 提供道路
        partner_obs=False,       # 同上
        norm_obs=False,
        reward_type="weighted_combination",
        dynamics_model="classic",
        collision_behavior="ignore",
        dist_to_goal_threshold=2.0,
        polyline_reduction_threshold=0.1,
        obs_radius=50.0,
        steer_actions=torch.linspace(-0.4, 0.4, 13),
        accel_actions=torch.linspace(-3.0, 1.5, 7),
        episode_len=args.max_steps,
    )
    env = GPUDriveTorchEnv(config=env_config, data_loader=data_loader,
                           max_cont_agents=args.max_agents, device=args.device)

    # 3. 场景 JSON 加载（用于 builder）
    scenes = []
    for fpath in env.data_batch:
        with open(fpath, 'r') as f:
            scenes.append(json.load(f))

    builder = GPUDriveObservationBuilder(env, scenes)

    # 4. 确定每个 world 的 ego index
    ego_indices = []
    for w in range(args.num_worlds):
        controllable = env.cont_agent_mask[w].nonzero(as_tuple=True)[0]
        ego_indices.append(controllable[0].item() if len(controllable) > 0 else 0)

    # 5. 初始化 agent
    agent = DiscreteIDCAgent(env, args, args.device)
    frames = {f"env_{i}": [] for i in range(args.num_worlds)}

    print(f'开始训练')
    # 6. 训练循环
    for epoch in range(args.epochs):
        obs = env.reset()
        # 重置 builder 的步数计数器
        for w in range(args.num_worlds):
            builder.reset_world_step(w, 0)

        for step in range(args.max_steps):
            # 收集所有 ego 的 IDC 观测
            states = []
            road_list = []
            for w in range(args.num_worlds):
                net, road, ref_raw, ref_err, other = builder.get_idc_observation(
                    w, ego_indices[w], perceived_distance=30.0)
                states.append(net)
                road_list.append(road)
                agent.road_states[w] = road      # 缓存，供后续约束计算

            states_batch = torch.tensor(np.stack(states), dtype=torch.float32, device=args.device)

            # 选动作
            action_phy, action_1d = agent.select_action(states_batch, deterministic=False)
            print(f'步数：{step},形状：{action_1d.shape}')

            # 构造 GPUDrive 所需的动作张量 [num_worlds, max_agents]
            act_template = torch.zeros((args.num_worlds, args.max_agents), dtype=torch.int64)
            for w in range(args.num_worlds):
                act_template[w, ego_indices[w]] = int(action_1d[w])
            env.step_dynamics(act_template)

            next_obs = env.get_obs()
            rewards = env.get_rewards()
            dones = env.get_dones()
            infos = env.get_infos()

            # 收集经验（需将 builder 的步数 +1）
            for w in range(args.num_worlds):
                builder.increment_step(w)
            # 注意：builder 步数递增后，再取观测会是下一步的，存经验时需保存 next_obs
            # 这里简化处理，实际应将 next_obs 转为 IDC 观测存入 buffer

            # 更新 agent（每收集一定步数后调用一次）
            # if step % args.update_freq == 0:
                # agent.update(...)   # 需要传入 ref_path 和 road_states_batch

            obs = next_obs

            if step % 5 == 0:
                imgs = env.vis.plot_simulator_state(
                    env_indices=list(range(args.num_worlds)),
                    time_steps=[epoch] * args.num_worlds,
                    zoom_radius=70,
                )
                for i in range(args.num_worlds):
                    frames[f"env_{i}"].append(img_from_fig(imgs[i]))

    # 每个环境的帧保存为单独的 GIF
    print("开始保存")
    save_dir = "/workspace/idc/gifs"
    os.makedirs(save_dir, exist_ok=True)

    for env_name, frame_list in frames.items():
        path = os.path.join(save_dir, f"rollout_{env_name}.gif")
        imageio.mimsave(path, frame_list, fps=5)
    print("Training finished.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--num-worlds', type=int, default=20)
    parser.add_argument('--max-agents', type=int, default=64)
    parser.add_argument('--max-steps', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dt', type=float, default=0.05)
    parser.add_argument('--horizon', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--lr-actor', type=float, default=3e-4)
    parser.add_argument('--lr-critic', type=float, default=3e-4)
    parser.add_argument('--init-penalty', type=float, default=1.0)
    parser.add_argument('--max-penalty', type=float, default=100.0)
    parser.add_argument('--amplifier-c', type=float, default=1.5)
    parser.add_argument('--amplifier-m', type=int, default=10)
    parser.add_argument('--update-freq', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    train(args)