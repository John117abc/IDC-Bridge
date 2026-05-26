import sys
import math
import numpy as np
import torch
import json
import os
import glob
import random
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

    # 自动检测全量数据池大小
    total_files = len([f for f in os.listdir(args.data_dir) if f.startswith('tfrecord')])
    dataset_size = args.dataset_size if args.dataset_size > 0 else total_files
    if dataset_size < args.num_worlds * 2:
        dataset_size = args.num_worlds * 2  # 最少保持 2x 的候选池
    logger.info(f'数据池: {dataset_size} 个文件, 训练时随机抽取 {args.num_worlds} 个世界')
    
    # 1. 数据加载器
    data_loader = SceneDataLoader(
        root=args.data_dir,
        batch_size=args.num_worlds,
        dataset_size=dataset_size,
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

    # 加载全量文件列表（供后续重采样用）
    all_files = sorted(glob.glob(os.path.join(args.data_dir, "tfrecord*")))
    all_files = all_files[:dataset_size]
    random.Random(args.seed).shuffle(all_files)
    logger.info(f'全量文件池: {len(all_files)} 个')

    # ================================================
    # 世界密度缓存：筛选高密度场景供 penalty 训练
    # ================================================
    density_cache_file = os.path.join(args.file_dir, "world_density.json")
    density_cache = {}
    if os.path.exists(density_cache_file):
        with open(density_cache_file, 'r') as f:
            density_cache = json.load(f)
        logger.info(f'密度缓存已加载: {len(density_cache)} 个世界')
    else:
        probe_size = min(2000, len(all_files))
        probe_files = random.sample(all_files, probe_size)
        logger.info(f'正在计算世界密度缓存（随机抽样 {probe_size}/{len(all_files)} 个文件，约 30 秒）...')
        first_batch = list(env.data_batch)
        batch_size = args.num_worlds
        for batch_start in range(0, probe_size, batch_size):
            batch_end = min(batch_start + batch_size, probe_size)
            batch = probe_files[batch_start:batch_end]
            if len(batch) < batch_size:
                batch = batch + [probe_files[0]] * (batch_size - len(batch))
            env.swap_data_batch(data_batch=batch)
            env.reset()
            p_np = env.sim.partner_observations_tensor().to_torch().cpu().numpy()
            for w in range(batch_size):
                fidx = batch_start + w
                if fidx < probe_size:
                    p = p_np[w, 0]
                    valid = (p[:, 0] != 0.0) | (p[:, 1] != 0.0) | (p[:, 2] != 0.0)
                    density_cache[probe_files[fidx]] = float(valid.sum())
            torch.cuda.empty_cache()
        # 恢复初始批次
        env.swap_data_batch(data_batch=first_batch)
        env.reset()
        torch.cuda.empty_cache()
        with open(density_cache_file, 'w') as f:
            json.dump(density_cache, f)
        logger.info(f'密度缓存已保存: {len(density_cache)} 个世界')

    # 构建稠密世界候选池
    dense_files = sorted([f for f, d in density_cache.items()
                          if d >= args.min_partner_density],
                         key=lambda f: density_cache[f], reverse=True)
    dense_files = dense_files[:args.dense_sample_size]
    logger.info(f'稠密世界池: {len(dense_files)} 个 (density >= {args.min_partner_density})')

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

    # --- 世界重采样函数 ---
    def resample_worlds(bad_worlds_set, ego_idx_list):
        logger.info(f'[RESAMPLE] Triggered: {len(bad_worlds_set)}/{args.num_worlds} bad worlds. Reloading...')
        
        # 0. 主动释放旧显存，防止 swap 期间新旧数据叠加 OOM
        for attr in ['ref_tensor', 'expert_pos', 'expert_vel', 'expert_heading', 'candidate_paths', '_road_cache']:
            if hasattr(builder, attr):
                delattr(builder, attr)
        torch.cuda.empty_cache()
        
        # 1. 抽取新世界（优先稠密池）
        candidate_pool = dense_files if len(dense_files) >= args.num_worlds * 2 else all_files
        batch_files = random.sample(candidate_pool, args.num_worlds)
        env.swap_data_batch(data_batch=batch_files)
        
        # 2. 更新 ego_indices
        new_ego = []
        for w in range(args.num_worlds):
            ctrl = env.cont_agent_mask[w].nonzero(as_tuple=True)[0]
            new_ego.append(ctrl[0].item() if len(ctrl) > 0 else 0)
        
        # 3. 重建 builder 数据
        builder._setup_expert_data()
        builder.clear_cache()
        builder.generate_candidate_paths(new_ego, num_paths=3)
        
        # 4. 阶段 1 路径坐标过滤
        bad_worlds_set.clear()
        for w in range(args.num_worlds):
            a = new_ego[w]
            for pid in range(builder.num_candidate_paths):
                pos = builder.candidate_paths[w][a][pid]['pos']
                if np.max(np.abs(pos[:, 0])) > 5000 or np.max(np.abs(pos[:, 1])) > 5000:
                    bad_worlds_set.add(w)
                    break
        
        # 5. 重置环境
        env.reset()
        for w in range(args.num_worlds):
            builder.reset_world_step(w, 0)
        builder.clear_cache()
        
        # 6. 更新 agent + 清 buffer
        agent.update_ego_indices(new_ego)
        agent.clear_buffer()
        
        logger.info(f'[RESAMPLE] complete: {len(bad_worlds_set)} bad, {args.num_worlds - len(bad_worlds_set)} good')
        return new_ego

    # 6. 训练循环
    for epoch in range(epochs):
        epoch += current_epoch
        obs = env.reset()
        for w in range(args.num_worlds):
            builder.reset_world_step(w, 0)
        builder.clear_cache()

        # 每个 epoch 独立记录损失
        epoch_history = []

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
                logger.info(f'[DIAG-ref] max pos_err={max_dp:.2f}m @world_{max_w} '
                            f'good_worlds={args.num_worlds - len(bad_worlds)}/{args.num_worlds}')

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
                    path = builder.candidate_paths[w][a][pid]
                    rx, ry = float(path['pos'][0, 0]), float(path['pos'][0, 1])
                    rh, rs = float(path['heading'][0]), float(path['speed'][0])
                    ref = np.array([rx, ry, rs, 0.0, rh, 0.0], dtype=np.float32)
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
                    epoch_history.append((critic_loss, actor_loss, agent.rho))

            # 记录帧
            # recorder.record(env, epoch, step)

            agent.global_step += 1
            # 暂时用不到
            # rewards = env.get_rewards()
            # dones = env.get_dones()
            # infos = env.get_infos()
        
        agent.globe_eps += 1  # 每个 epoch 结束后增加全局回合数

        # 保存本 epoch 损失记录
        agent.history_loss.append(epoch_history.copy())
        
        logger.info(f'开始保存轨迹图像')
        for viz in viz_list:
            if len(viz.actual_x) > 0:
                viz.save_plot(viz_dir, epoch + 1)

        # 保存模型
        save_info = {
                'env_name': 'examples',
        }
        agent.save(save_info=save_info)

        # === 坏世界超过阈值时触发世界重采样 ===
        if len(bad_worlds) > args.max_bad_worlds:
            logger.warning(f'[RESAMPLE] bad={len(bad_worlds)}/{args.num_worlds}, good<{args.num_worlds - args.max_bad_worlds}, triggering resample...')
            ego_indices = resample_worlds(bad_worlds, ego_indices)

    # 每个环境的帧保存为单独的 GIF
    # recorder.save_all_gifs()

    logger.debug('训练完成')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--num-worlds', type=int, default=150)
    parser.add_argument('--max-agents', type=int, default=1)
    parser.add_argument('--max-steps', type=int, default=90)
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dt', type=float, default=0.1)
    parser.add_argument('--horizon', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--lr-actor', type=float, default=8e-5)
    parser.add_argument('--lr-critic', type=float, default=3e-4)
    parser.add_argument('--init-penalty', type=float, default=1.0,
                        help='初始 ρ，0=纯追踪模式')
    parser.add_argument('--max-penalty', type=float, default=20.0)
    parser.add_argument('--amplifier-c', type=float, default=1.015)
    parser.add_argument('--pim-interval', type=int, default=30)
    parser.add_argument('--dataset-size', type=int, default=0,
                        help='全量数据池大小，0=自动检测 data_dir 下的 tfrecord 文件数')
    parser.add_argument('--max-bad-worlds', type=int, default=50,
                        help='坏世界数超过此值时触发世界重采样')
    parser.add_argument('--min-partner-density', type=float, default=2.0,
                        help='稠密世界筛选阈值：平均周车数低于此值的世界不进入候选池')
    parser.add_argument('--dense-sample-size', type=int, default=500,
                        help='稠密世界候选池大小')
    parser.add_argument('--fix-speed', action='store_true', default=False,
                        help='[诊断] speed_err恒为0，排除速度干扰')
    parser.add_argument('--fix-heading', action='store_true', default=False,
                        help='[诊断] heading_err恒为0，排除朝向干扰')
    parser.add_argument('--no-sign', action='store_true', default=False,
                        help='[诊断] delta_p不去符号，排除符号翻转问题')
    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--save-freq', type=int, default=5)
    parser.add_argument('--file-dir', type=str, default="/workspace/data")
    parser.add_argument('--load-model', type=bool, default=True)
    parser.add_argument('--model-path', type=str, default="/workspace/data/checkpoints/20260526/idc-waymo-v1.0_examples_152154_episode=5.pth")
    args = parser.parse_args()
    train(args)