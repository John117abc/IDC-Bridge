import sys
import numpy as np
import torch
import imageio
import json
import os
# 导入智能体


# 导入 GPUDrive 相关
from gpudrive.env.config import EnvConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.visualize.utils import img_from_fig

# 导入已完成的观测构建器
sys.path.insert(0, '/workspace/idc/src')
from env.idc_state_builder import GPUDriveObservationBuilder
from agents.idc_agent import DiscreteIDCAgent
from buffer import PERBuffer
from utils import get_logger
logger = get_logger('idc-agent')

# 方式2：多维动作 (num_worlds, max_agents, action_dim)
def create_forward_actions_multidim(num_worlds, max_agents, action_dim=3):
    """
    创建三维动作张量
    形状: (num_worlds, max_agents, action_dim)
    """
    # 向前走的动作：[转向=0, 加速度=0.5, 头部倾斜=0]
    forward_action = torch.tensor([0.5, 0.0, 0.0], dtype=torch.float32)
    
    # 扩展到所有智能体
    actions = forward_action.reshape(1, 1, -1).expand(num_worlds, max_agents, -1).clone()
    return actions

def extend_action_to_3d(actions_2d):
    """
    将二维动作 [delta, a] 扩展为三维动作 [delta, a, 0]
    
    参数:
        actions_2d (torch.Tensor): 形状 (batch, max_agents, 2) 的动作张量
    返回:
        torch.Tensor: 形状 (batch, max_agents, 3) 的动作张量，第三维恒为 0
    """
    batch, agents, _ = actions_2d.shape
    # 创建形状匹配的零张量作为第三维
    zeros = torch.zeros((batch, agents, 1), dtype=actions_2d.dtype, device=actions_2d.device)
    # 沿最后一个维度拼接
    actions_3d = torch.cat([actions_2d, zeros], dim=-1)
    return actions_3d

# ==============================================
# 训练主循环
# ==============================================
def train(args):
    
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
        road_map_obs=True,
        partner_obs=True,
        norm_obs=True,
        reward_type="weighted_combination",
        dynamics_model="classic",
        collision_behavior="ignore",
        max_controlled_agents=1,
        dist_to_goal_threshold=2.0,
        polyline_reduction_threshold=0.1,
        obs_radius=50.0,
        episode_len=args.max_steps,
        
    )
    
    env = GPUDriveTorchEnv(
        config=env_config,
        data_loader=data_loader,
        max_cont_agents=args.max_agents, # Maximum number of agents to control per scenario
        device=args.device,
        action_type="continuous",
)

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
    agent = DiscreteIDCAgent(env, args, args.device, builder,ego_indices)

    print(f'开始训练')
    # 6. 训练循环
    for epoch in range(args.epochs):
        obs = env.reset()
        # 重置 builder 的步数计数器
        for w in range(args.num_worlds):
            builder.reset_world_step(w, 0)

        for step in range(args.max_steps):
            # 记录状态，供后续动作选择和训练使用
            logger.info(f'回合 {epoch+1}/{args.epochs}, 步数 {step+1}/{args.max_steps}')
            states = []
            for w in range(args.num_worlds):
                network_state, raw_state = builder.get_idc_observation(
                    w, ego_indices[w])
                states.append(network_state)
                agent.buffer.handle_new_experience((network_state, raw_state, w))  # 将原始状态也存入 buffer
                # 增加对应世界的步数计数器
                builder.increment_step(w)
            logger.info(f'状态构建完成，开始选择动作')
            # 创建批量动作
            # actions = create_forward_actions_multidim(args.num_worlds, args.max_agents, action_dim=3)
            actions = agent.select_action(states)  # [num_worlds, max_agents, action_dim]
            actions = extend_action_to_3d(actions)
            logger.info(f'动作选择完成，开始环境交互，动作形状: {actions.shape}')
            env.step_dynamics(actions)

            logger.info(f'环境交互完成，开始更新智能体')
            # 更新参数
            agent.update()
            next_obs = env.get_obs()
            obs = next_obs
            agent.global_step += 1
            # 暂时用不到
            # rewards = env.get_rewards()
            # dones = env.get_dones()
            # infos = env.get_infos()

    # 每个环境的帧保存为单独的 GIF
    print("结束")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--num-worlds', type=int, default=10)
    parser.add_argument('--max-agents', type=int, default=1)
    parser.add_argument('--max-steps', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dt', type=float, default=0.05)
    parser.add_argument('--horizon', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--lr-actor', type=float, default=3e-4)
    parser.add_argument('--lr-critic', type=float, default=3e-4)
    parser.add_argument('--init-penalty', type=float, default=1.0)
    parser.add_argument('--max-penalty', type=float, default=100.0)
    parser.add_argument('--amplifier-c', type=float, default=1.5)
    parser.add_argument('--amplifier-m', type=int, default=10)
    parser.add_argument('--update-freq', type=int, default=5)
    parser.add_argument('--seed', type=int, default=20)
    args = parser.parse_args()
    train(args)