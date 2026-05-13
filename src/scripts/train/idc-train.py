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
    agent = DiscreteIDCAgent(env, args, args.device)

    print(f'开始训练')
    # 6. 训练循环
    for epoch in range(args.epochs):
        obs = env.reset()
        # 重置 builder 的步数计数器
        for w in range(args.num_worlds):
            builder.reset_world_step(w, 0)

        logger.info(f'观察空间形状: {obs.shape}, 动作空间: {env.action_space}')

        for step in range(args.max_steps):
            # 收集所有 ego 的 IDC 观测
            states = []
            road_list = []
            
            for w in range(args.num_worlds):
                net, road, ref, ref_err, other = builder.get_idc_observation(
                    w, ego_indices[w], perceived_distance=30.0)
                states.append(net)
                road_list.append(road)
                agent.road_states[w] = road      # 缓存，供后续约束计算

            states_batch = torch.tensor(np.stack(states), dtype=torch.float32, device=args.device)

            # 选动作
            _, action_1d = agent.select_action(states_batch, deterministic=False)


            # 构造 GPUDrive 所需的动作张量 [num_worlds, max_agents]
            act_template = torch.zeros((args.num_worlds, args.max_agents), dtype=torch.int64)
            for w in range(args.num_worlds):
                act_template[w, ego_indices[w]] = int(action_1d[w])
            env.step_dynamics(act_template)

            next_obs = env.get_obs()
            print(f'下一步观测形状: {next_obs.shape}')
            rewards = env.get_rewards()
            dones = env.get_dones()
            infos = env.get_infos()
            print(f'奖励形状: {rewards.shape}, done 形状: {dones.shape}')
            print("infos type:", type(infos))
            if hasattr(infos, 'shape'):
                info_dict = dict(infos)
                print("infos 形状:", infos.shape)
                print("infos[0,0]", infos[0,0].cpu().numpy())   # 第一个 world 第一个 agent
            else:
                # 如果是 list of dicts
                print("infos sample:", infos[0])
            # 收集经验（需将 builder 的步数 +1）
            for w in range(args.num_worlds):
                builder.increment_step(w)
            # 注意：builder 步数递增后，再取观测会是下一步的，存经验时需保存 next_obs
            # 这里简化处理，实际应将 next_obs 转为 IDC 观测存入 buffer

            # 更新 agent（每收集一定步数后调用一次）
            # if step % args.update_freq == 0:
                # agent.update(...)   # 需要传入 ref_path 和 road_states_batch

            obs = next_obs


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
    parser.add_argument('--update-freq', type=int, default=10)
    parser.add_argument('--seed', type=int, default=20)
    args = parser.parse_args()
    train(args)