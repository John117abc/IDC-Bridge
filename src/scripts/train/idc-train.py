import sys
import math
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
        sample_with_replacement=False,  # 每轮从全量数据中不重复采样
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
    max_step = builder.EXPERT_TRAJ_LEN if builder.EXPERT_TRAJ_LEN is not None else 91
    ego_indices = []
    for w in range(args.num_worlds):
        controllable = env.cont_agent_mask[w].nonzero(as_tuple=True)[0]
        ego_indices.append(controllable[0].item() if len(controllable) > 0 else 0)

    builder.generate_candidate_paths(ego_indices, num_paths=3)

    # 5. 初始化 agent
    agent = DiscreteIDCAgent(env, args, args.device, builder,ego_indices)

    epochs = args.epochs
    current_epoch = 0
    # 判断是否需要加载模型
    if args.load_model:
        logger.info(f'正在加载模型: {args.model_path}')
        agent.load(args.model_path)
        current_epoch = agent.globe_eps
        epochs -= current_epoch

    logger.info(f'训练开始: epochs={args.epochs}, num_worlds={args.num_worlds}, max_steps={max_step}')

    viz_dir = os.path.join(args.file_dir, 'traj_plots')
    os.makedirs(viz_dir, exist_ok=True)

    # 历史损失
    history_loss = []

    # --- 预训练世界过滤 ---
    bad_worlds = set()

    # 阶段1：路径坐标异常检测
    logger.info('Pre-training world filter: checking path coordinates...')
    for w in range(args.num_worlds):
        a = ego_indices[w]
        for pid in range(builder.num_candidate_paths):
            pos = builder.candidate_paths[w][a][pid]['pos']
            if np.max(np.abs(pos[:, 0])) > 5000 or np.max(np.abs(pos[:, 1])) > 5000:
                bad_worlds.add(w)
                logger.warning(f'[FILTER-path] world_{w} path_{pid} max|pos|=({np.max(np.abs(pos[:,0])):.0f},{np.max(np.abs(pos[:,1])):.0f})')
                break

    logger.info(f'[FILTER] {len(bad_worlds)}/{args.num_worlds} worlds excluded (path anomaly)')

    # 6. 训练循环
    for epoch in range(epochs):
        epoch += current_epoch
        obs = env.reset()
        for w in range(args.num_worlds):
            builder.reset_world_step(w, 0)
        builder.clear_cache()

        # 可视化：每 epoch 从当前未拉黑世界中选前 10 个画轨迹
        VIZ_WORLDS = [w for w in range(args.num_worlds) if w not in bad_worlds][:10]
        viz_list = [TrajectoryVisualizer(builder, w, ego_indices[w])
                    for w in VIZ_WORLDS]

        # 候选路径索引：每个 episode 固定，避免步间跳变
        episode_path_indices = [np.random.randint(builder.num_candidate_paths)
                                for _ in range(args.num_worlds)]

        for step in range(max_step):
            if step % 10 == 0:
                logger.info(f'回合 {epoch+1}/{args.epochs}, 步数 {step+1}/{max_step}')
            else:
                logger.debug(f'回合 {epoch+1}/{args.epochs}, 步数 {step+1}/{max_step}')

            states = builder.get_idc_observations_batch(ego_indices,
                                                        path_indices=episode_path_indices)

            # [诊断] 每步剔除 ego 坐标异常 world，同时打印 delta_p 判断是否为误杀
            ref_start = agent.DIM_EGO + agent.DIM_OTHERS + agent.DIM_VALIDITY
            for w in range(args.num_worlds):
                if w in bad_worlds:
                    continue
                if abs(states[w][0]) > 5000 or abs(states[w][1]) > 5000:
                    bad_worlds.add(w)
                    dp = abs(states[w][ref_start])
                    logger.warning(f'[FILTER-ego] world_{w} step={step} ego=({states[w][0]:.0f},{states[w][1]:.0f}) delta_p={dp:.0f}')

            # [诊断] --no-sign: 去掉初始状态 delta_p 符号
            if args.no_sign:
                ref_start = agent.DIM_EGO + agent.DIM_OTHERS + agent.DIM_VALIDITY
                for w in range(args.num_worlds):
                    if w in bad_worlds:
                        continue
                    states[w][ref_start] = abs(states[w][ref_start])

            # [DIAG] 参考误差: 哪个 world 偏离最大
            if step % 5 == 0:
                max_dp, max_w = 0.0, 0
                for w in range(args.num_worlds):
                    if w in bad_worlds:
                        continue
                    s = states[w]
                    ref_start = agent.DIM_EGO + agent.DIM_OTHERS + agent.DIM_VALIDITY
                    dp, dphi, dv = abs(s[ref_start]), abs(s[ref_start+1]), abs(s[ref_start+2])
                    if dp > max_dp:
                        max_dp, max_w = dp, w
                logger.info(f'[DIAG-ref] max pos_err={max_dp:.2f}m @world_{max_w}')

            for w in range(args.num_worlds):
                if w in bad_worlds:
                    continue
                agent.buffer.handle_new_experience((states[w], w, episode_path_indices[w]))
                builder.increment_step(w)

            # 记录可视化世界的自车位置
            positions = builder.get_ego_positions_batch(ego_indices)
            for i, w in enumerate(VIZ_WORLDS):
                if w in bad_worlds:
                    continue
                viz_list[i].record_step(positions[w, 0], positions[w, 1])

            logger.debug(f'状态构建完成，开始选择动作')
            # 创建批量动作
            actions = agent.select_action(states)  # [num_worlds, max_agents, action_dim]

            # [DIAG] 第一步打印初始速度和对应动作
            if step == 0:
                state_tensor = agent.batch_state_to_tensor(states[:2])
                with torch.no_grad():
                    norm_action_raw = agent.actor(state_tensor)  # tanh 前 [-1,1]
                    norm_2d = norm_action_raw.view(2, 1, 2)
                for i in [0, 1]:
                    init_speed = float(states[i][2])
                    acc = actions[i, 0, 0].item()
                    steer = actions[i, 0, 1].item()
                    rsteer = norm_2d[i, 0, 0].item()
                    racc = norm_2d[i, 0, 1].item()
                    w = i
                    a = ego_indices[w]
                    pid = episode_path_indices[w]
                    ref_spd0 = builder.candidate_paths[w][a][pid]['speed'][0]
                    # utility 分解
                    ego = np.array([float(states[i][j]) for j in range(6)])
                    ref = builder.get_ref_state_from_path(w, a, pid, ego[0], ego[1])
                    pos_err = np.hypot(ego[0] - ref[0], ego[1] - ref[1])
                    heading_err = ego[4] - ref[4]
                    speed_err = np.hypot(ego[2], ego[3]) - ref[2]
                    logger.debug(f'[DIAG-init] world_{i} speed={init_speed:.2f} acc={acc:.3f} steer={steer:.3f} '
                                f'norm_acc={racc:.4f} ref_spd={ref_spd0:.2f} '
                                f'pos_err={pos_err:.2f} heading_err={heading_err:.3f} speed_err={speed_err:.2f}')

            actions = extend_action_to_3d(actions)
            logger.debug(f'动作选择完成，开始环境交互，动作形状: {actions.shape}')
            env.step_dynamics(actions)

            # 检测极端位置 teleport (15575m级跳变)
            if step % 5 == 0:
                abs_np = builder.sim.absolute_self_observation_tensor().to_torch().cpu().numpy()
                for w in range(args.num_worlds):
                    a = ego_indices[w]
                    x, y = float(abs_np[w, a, 0]), float(abs_np[w, a, 1])
                    if abs(x) > 1e5 or abs(y) > 1e5:
                        logger.warning(f'[TELEPORT] world_{w} step={step} x={x:.1f} y={y:.1f}')

            logger.debug(f'环境交互完成，开始更新智能体')
            # 更新参数
            if agent.buffer.should_start_training():
                logger.debug(f'开始训练: global_step={agent.global_step}, buffer size={len(agent.buffer)}')
                critic_loss, actor_loss = agent.update()
                logger.debug(f'训练完成: global_step={agent.global_step}')

                # 打印损失
                if actor_loss is not None:
                    logger.info(f'Critic Loss: {critic_loss:.4f}, Actor Loss: {actor_loss ==None and "N/A" or f"{actor_loss:.4f}" }, Penalty Coefficient (rho): {agent.rho:.4f}')
                    history_loss.append((critic_loss, actor_loss, agent.rho))

            # 记录帧
            # recorder.record(env, epoch, step)

            agent.global_step += 1
            # 暂时用不到
            # rewards = env.get_rewards()
            # dones = env.get_dones()
            # infos = env.get_infos()
        
        agent.globe_eps += 1  # 每个 epoch 结束后增加全局回合数
        
        logger.info(f'开始保存轨迹图像')
        for viz in viz_list:
            if len(viz.actual_x) > 0:
                viz.save_plot(viz_dir, epoch + 1)

        # 保存模型
        save_info = {
                'history_loss':history_loss,
                'env_name': 'examples',
        }
        agent.save(save_info=save_info)

    # 每个环境的帧保存为单独的 GIF
    # recorder.save_all_gifs()

    logger.debug('训练完成')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--num-worlds', type=int, default=200)
    parser.add_argument('--max-agents', type=int, default=1)
    parser.add_argument('--max-steps', type=int, default=90)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dt', type=float, default=0.1)
    parser.add_argument('--horizon', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--lr-actor', type=float, default=8e-5)
    parser.add_argument('--lr-critic', type=float, default=3e-4)
    parser.add_argument('--init-penalty', type=float, default=1.0)
    parser.add_argument('--max-penalty', type=float, default=10.0)
    parser.add_argument('--amplifier-c', type=float, default=1.015)
    parser.add_argument('--pim-interval', type=int, default=30)
    parser.add_argument('--tracking-only', action='store_true', default=False,
                        help='纯跟踪诊断模式: 无penalty/无噪声/ρ固定为0')
    parser.add_argument('--fix-speed', action='store_true', default=False,
                        help='[诊断] speed_err恒为0，排除速度干扰')
    parser.add_argument('--fix-heading', action='store_true', default=False,
                        help='[诊断] heading_err恒为0，排除朝向干扰')
    parser.add_argument('--no-sign', action='store_true', default=False,
                        help='[诊断] delta_p不去符号，排除符号翻转问题')
    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--save-freq', type=int, default=5)
    parser.add_argument('--file-dir', type=str, default="/workspace/data")
    parser.add_argument('--load-model', type=bool, default=False)
    parser.add_argument('--model-path', type=str, default="/workspace/data/checkpoints/20260519/idc-waymo-v1.0_examples_150035_episode=50.pth")
    args = parser.parse_args()
    train(args)