import sys
import numpy as np
import torch
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
from utils import get_logger, VisualRecorder, TrajectoryVisualizer
logger = get_logger('idc-agent')

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
# 评估主循环
# ==============================================
def evaluate(args):
    
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
        # episode_len=args.max_steps,
    )
    
    env = GPUDriveTorchEnv(
        config=env_config,
        data_loader=data_loader,
        max_cont_agents=args.max_agents, # Maximum number of agents to control per scenario
        device=args.device,
        action_type="continuous",
    )

    # 初始化记录器
    recorder = VisualRecorder(
        num_worlds=args.num_worlds,
        save_dir="/workspace/data/gifs",
        fps=5
    )

    # 3. 场景 JSON 加载（用于 builder）
    scenes = []
    for fpath in env.data_batch:
        with open(fpath, 'r') as f:
            scenes.append(json.load(f))

    builder = GPUDriveObservationBuilder(env, scenes)
    max_step = builder.EXPERT_TRAJ_LEN
    # 4. 确定每个 world 的 ego index
    ego_indices = []
    for w in range(args.num_worlds):
        controllable = env.cont_agent_mask[w].nonzero(as_tuple=True)[0]
        ego_indices.append(controllable[0].item() if len(controllable) > 0 else 0)

    builder.generate_candidate_paths(ego_indices, num_paths=3)

    # 5. 初始化 agent
    agent = DiscreteIDCAgent(env, args, args.device, builder,ego_indices)

    # 加载模型
    logger.info(f'正在加载模型: {args.model_path}')
    agent.load(args.model_path)

    logger.info(f'评估开始: epochs={args.epochs}, num_worlds={args.num_worlds}, max_steps={max_step}')

    viz_dir = os.path.join(args.file_dir, 'eval_plots')
    os.makedirs(viz_dir, exist_ok=True)

    # 6. 评估循环
    for epoch in range(args.epochs):
        obs = env.reset()
        for w in range(args.num_worlds):
            builder.reset_world_step(w, 0)

        viz_list = [TrajectoryVisualizer(builder, w, ego_indices[w])
                    for w in range(args.num_worlds)]

        for step in range(max_step):
            logger.info(f'回合 {epoch+1}/{args.epochs}, 步数 {step+1}/{max_step}')

            # 评估用中心路径 (path_idx=0)
            path_indices = [0 for _ in range(args.num_worlds)]
            states = builder.get_idc_observations_batch(
                ego_indices, path_indices=path_indices)
            for w in range(args.num_worlds):
                builder.increment_step(w)

            # 记录自车位置
            positions = builder.get_ego_positions_batch(ego_indices)
            for i in range(args.num_worlds):
                viz_list[i].record_step(positions[i, 0], positions[i, 1])

            # 确定性的动作选择（不添加探索噪声）
            actions = agent.select_action(states, deterministic=True)
            actions = extend_action_to_3d(actions)
            logger.debug(f'动作选择完成，开始环境交互，动作形状: {actions.shape}')
            env.step_dynamics(actions)

            logger.debug(f'环境交互完成')

            # 记录帧
            recorder.record(env, epoch, step)

        for viz in viz_list:
            if len(viz.actual_x) > 0:
                viz.save_plot(viz_dir, epoch + 1)

    # 每个环境的帧保存为单独的 GIF
    recorder.save_all_gifs()

    logger.debug('评估完成')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--num-worlds', type=int, default=2)
    parser.add_argument('--max-agents', type=int, default=1)
    parser.add_argument('--max-steps', type=int, default=90)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dt', type=float, default=0.1)
    parser.add_argument('--horizon', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--lr-actor', type=float, default=1e-5)
    parser.add_argument('--lr-critic', type=float, default=3e-4)
    parser.add_argument('--init-penalty', type=float, default=1.0)
    parser.add_argument('--max-penalty', type=float, default=100.0)
    parser.add_argument('--amplifier-c', type=float, default=1.005)
    parser.add_argument('--pim-interval', type=int, default=200)
    parser.add_argument('--seed', type=int, default=20)
    parser.add_argument('--save-freq', type=int, default=1)
    parser.add_argument('--file-dir', type=str, default="/workspace/data")
    parser.add_argument('--load-model', type=bool, default=False)
    parser.add_argument('--model-path', type=str, default="/workspace/data/checkpoints/20260516/idc-waymo-v1.0_examples_143243_episode=1.pth")
    args = parser.parse_args()
    evaluate(args)