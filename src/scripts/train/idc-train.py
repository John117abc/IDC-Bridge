import sys
import os
import glob
import random

import numpy as np
import torch

from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.dataset import SceneDataLoader

sys.path.insert(0, '/workspace/idc/src')
from env.env_utils import extend_action_to_3d, get_env_config, load_scenes, get_ego_indices
from env.world_manager import WorldManager
from env.idc_state_builder import GPUDriveObservationBuilder
from agents.idc_agent import DiscreteIDCAgent
from utils import get_logger, VisualRecorder, TrajectoryVisualizer

logger = get_logger('idc-agent')


def _diag_init_step(args, agent, builder, ego_indices, episode_path_indices,
                    states, actions, step):
    if step != 0:
        return

    state_tensor = agent.batch_state_to_tensor(states[:2])
    with torch.no_grad():
        norm_action_raw = agent.actor(state_tensor)
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
        ego = np.array([float(states[i][j]) for j in range(6)])
        path = builder.candidate_paths[w][a][pid]
        rx, ry = float(path['pos'][0, 0]), float(path['pos'][0, 1])
        rh, rs = float(path['heading'][0]), float(path['speed'][0])
        ref = np.array([rx, ry, rs, 0.0, rh, 0.0], dtype=np.float32)
        pos_err = np.hypot(ego[0] - ref[0], ego[1] - ref[1])
        heading_err = ego[4] - ref[4]
        speed_err = np.hypot(ego[2], ego[3]) - ref[2]
        logger.debug(
            f'[DIAG-init] world_{i} speed={init_speed:.2f} acc={acc:.3f} steer={steer:.3f} '
            f'norm_acc={racc:.4f} ref_spd={ref_spd0:.2f} '
            f'pos_err={pos_err:.2f} heading_err={heading_err:.3f} speed_err={speed_err:.2f}'
        )


def train(args):
    print("Data root:", args.data_dir)

    total_files = len([f for f in os.listdir(args.data_dir) if f.startswith('tfrecord')])
    dataset_size = args.dataset_size if args.dataset_size > 0 else total_files
    if dataset_size < args.num_worlds * 2:
        dataset_size = args.num_worlds * 2
    logger.info(f'数据池: {dataset_size} 个文件, 训练时随机抽取 {args.num_worlds} 个世界')

    data_loader = SceneDataLoader(
        root=args.data_dir,
        batch_size=args.num_worlds,
        dataset_size=dataset_size,
        sample_with_replacement=False,
        shuffle=True,
        seed=args.seed,
    )

    env_config = get_env_config()
    env = GPUDriveTorchEnv(
        config=env_config,
        data_loader=data_loader,
        max_cont_agents=args.max_agents,
        device=args.device,
        action_type="continuous",
    )

    all_files = sorted(glob.glob(os.path.join(args.data_dir, "tfrecord*")))
    all_files = all_files[:dataset_size]
    random.Random(args.seed).shuffle(all_files)
    logger.info(f'全量文件池: {len(all_files)} 个')

    recorder = VisualRecorder(
        num_worlds=args.num_worlds,
        save_dir="/workspace/data/gifs",
        fps=5,
    )

    scenes = load_scenes(env)
    builder = GPUDriveObservationBuilder(env, scenes)
    max_step = builder.EXPERT_TRAJ_LEN if builder.EXPERT_TRAJ_LEN is not None else 91

    ego_indices = get_ego_indices(env, args.num_worlds)
    builder.generate_candidate_paths(ego_indices, num_paths=3)

    agent = DiscreteIDCAgent(env, args, args.device, builder, ego_indices)

    wm = WorldManager(env, builder, agent, all_files, args, logger)
    wm.filter_initial(ego_indices)

    epochs = args.epochs
    current_epoch = 0
    if args.load_model:
        logger.info(f'正在加载模型: {args.model_path}')
        agent.load(args.model_path)
        current_epoch = agent.globe_eps
        epochs -= current_epoch

    logger.info(f'训练开始: epochs={args.epochs}, num_worlds={args.num_worlds}, max_steps={max_step}')

    viz_dir = os.path.join(args.file_dir, 'traj_plots')
    os.makedirs(viz_dir, exist_ok=True)

    for epoch in range(epochs):
        epoch += current_epoch
        obs = env.reset()
        for w in range(args.num_worlds):
            builder.reset_world_step(w, 0)
        builder.clear_cache()

        epoch_history = []

        VIZ_WORLDS = wm.good_worlds[:10]
        viz_list = [TrajectoryVisualizer(builder, w, ego_indices[w])
                    for w in VIZ_WORLDS]

        episode_path_indices = [np.random.randint(builder.num_candidate_paths)
                                for _ in range(args.num_worlds)]

        for step in range(max_step):
            if step % 10 == 0:
                logger.info(f'回合 {epoch+1}/{args.epochs}, 步数 {step+1}/{max_step}')
            else:
                logger.debug(f'回合 {epoch+1}/{args.epochs}, 步数 {step+1}/{max_step}')

            states = builder.get_idc_observations_batch(ego_indices,
                                                        path_indices=episode_path_indices)

            wm.filter_per_step(states, step)

            if args.no_sign:
                ref_start = agent.DIM_EGO + agent.DIM_OTHERS + agent.DIM_VALIDITY
                for w in range(args.num_worlds):
                    if w in wm.bad_worlds:
                        continue
                    states[w][ref_start] = abs(states[w][ref_start])

            for w in range(args.num_worlds):
                if w in wm.bad_worlds:
                    continue
                agent.buffer.handle_new_experience((states[w], w, episode_path_indices[w]))
                builder.increment_step(w)

            positions = builder.get_ego_positions_batch(ego_indices)
            for i, w in enumerate(VIZ_WORLDS):
                if w in wm.bad_worlds:
                    continue
                viz_list[i].record_step(positions[w, 0], positions[w, 1])

            logger.debug(f'状态构建完成，开始选择动作')
            actions = agent.select_action(states)

            _diag_init_step(args, agent, builder, ego_indices, episode_path_indices,
                            states, actions, step)

            actions = extend_action_to_3d(actions)
            logger.debug(f'动作选择完成，开始环境交互，动作形状: {actions.shape}')
            env.step_dynamics(actions)

            if step % 5 == 0:
                abs_np = builder.sim.absolute_self_observation_tensor().to_torch().cpu().numpy()
                for w in range(args.num_worlds):
                    a = ego_indices[w]
                    x, y = float(abs_np[w, a, 0]), float(abs_np[w, a, 1])
                    if abs(x) > 1e5 or abs(y) > 1e5:
                        logger.warning(f'[TELEPORT] world_{w} step={step} x={x:.1f} y={y:.1f}')

            logger.debug(f'环境交互完成，开始更新智能体')
            if agent.buffer.should_start_training():
                logger.debug(f'开始训练: global_step={agent.global_step}, buffer size={len(agent.buffer)}')
                critic_loss, actor_loss = agent.update()
                logger.debug(f'训练完成: global_step={agent.global_step}')

                if actor_loss is not None:
                    logger.info(f'Critic Loss: {critic_loss:.4f}, Actor Loss: {actor_loss:.4f}, '
                                f'Penalty Coefficient (rho): {agent.rho:.4f}')
                    epoch_history.append((critic_loss, actor_loss, agent.rho))

            agent.global_step += 1

        agent.globe_eps += 1
        agent.history_loss.append(epoch_history.copy())

        logger.info(f'开始保存轨迹图像')
        for viz in viz_list:
            if len(viz.actual_x) > 0:
                viz.save_plot(viz_dir, epoch + 1)

        save_info = {'env_name': 'examples'}
        agent.save(save_info=save_info)

        if wm.should_resample():
            logger.warning(
                f'[RESAMPLE] bad={len(wm.bad_worlds)}/{args.num_worlds}, '
                f'good<{args.num_worlds - args.max_bad_worlds}, triggering resample...'
            )
            ego_indices = wm.resample()

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
    parser.add_argument('--max-bad-worlds', type=int, default=100,
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
    parser.add_argument('--model-path', type=str,
                        default="/workspace/data/checkpoints/20260526/idc-waymo-v1.0_examples_153857_episode=15.pth")
    args = parser.parse_args()
    train(args)
