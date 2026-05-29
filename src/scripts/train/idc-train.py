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
from utils.config import build_config
from metrics import PDMSScorer

logger = get_logger('idc-train')


def _diag_init_step(config, agent, builder, ego_indices, episode_path_indices,
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


def train(config):
    print("Data root:", config.data_dir)

    total_files = len([f for f in os.listdir(config.data_dir) if f.startswith('tfrecord')])
    dataset_size = config.dataset_size if config.dataset_size > 0 else total_files
    if dataset_size < config.num_worlds * 2:
        dataset_size = config.num_worlds * 2
    logger.info(f'数据池: {dataset_size} 个文件, 训练时随机抽取 {config.num_worlds} 个世界')

    data_loader = SceneDataLoader(
        root=config.data_dir,
        batch_size=config.num_worlds,
        dataset_size=dataset_size,
        sample_with_replacement=False,
        shuffle=True,
        seed=config.seed,
    )

    env_config = get_env_config()
    env = GPUDriveTorchEnv(
        config=env_config,
        data_loader=data_loader,
        max_cont_agents=config.max_agents,
        device=config.device,
        action_type="continuous",
    )

    all_files = sorted(glob.glob(os.path.join(config.data_dir, "tfrecord*")))
    all_files = all_files[:dataset_size]
    random.Random(config.seed).shuffle(all_files)
    logger.info(f'全量文件池: {len(all_files)} 个')

    gif_enabled = getattr(config, 'gif_enabled', False)
    recorder = VisualRecorder(
        num_worlds=config.num_worlds,
        save_dir=config.gif_save_dir,
        fps=config.gif_fps,
    )

    scenes = load_scenes(env)
    builder = GPUDriveObservationBuilder(env, scenes)
    max_step = builder.EXPERT_TRAJ_LEN if builder.EXPERT_TRAJ_LEN is not None else 91

    ego_indices = get_ego_indices(env, config.num_worlds)
    builder.generate_candidate_paths(ego_indices, num_paths=3)

    agent = DiscreteIDCAgent(env, config, config.device, builder, ego_indices)

    wm = WorldManager(env, builder, agent, all_files, config, logger)
    wm.filter_initial(ego_indices)

    epochs = config.epochs
    current_epoch = 0
    if config.load_model and config.model_path:
        logger.info(f'正在加载模型: {config.model_path}')
        agent.load(config.model_path)
        current_epoch = agent.globe_eps
        epochs -= current_epoch

    logger.info(f'训练开始: epochs={config.epochs}, num_worlds={config.num_worlds}, max_steps={max_step}')

    viz_dir = os.path.join(config.file_dir, 'traj_plots')
    os.makedirs(viz_dir, exist_ok=True)

    for epoch in range(epochs):
        epoch += current_epoch
        obs = env.reset()
        for w in range(config.num_worlds):
            builder.reset_world_step(w, 0)
        builder.clear_cache()
        wm.reached_worlds.clear()

        # 检测 epoch 开头已 done 的世界（轨迹数据异常/已到终点）
        done_np = env.get_dones().cpu().numpy()
        for w in range(config.num_worlds):
            if w in wm.bad_worlds:
                continue
            a = ego_indices[w]
            if done_np[w, a] > 0.5:
                wm.reached_worlds.add(w)

        epoch_history = []

        VIZ_WORLDS = wm.good_worlds[:10]
        viz_list = [TrajectoryVisualizer(builder, w, ego_indices[w])
                    for w in VIZ_WORLDS]

        episode_path_indices = [np.random.randint(builder.num_candidate_paths)
                                for _ in range(config.num_worlds)]

        scorers = [PDMSScorer(config) for _ in range(config.num_worlds)]
        ref_start_agent = agent.DIM_EGO + agent.DIM_OTHERS + agent.DIM_VALIDITY

        for step in range(max_step):
            if step % 10 == 0:
                logger.info(f'回合 {epoch+1}/{config.epochs}, 步数 {step+1}/{max_step}')
            else:
                logger.debug(f'回合 {epoch+1}/{config.epochs}, 步数 {step+1}/{max_step}')

            states = builder.get_idc_observations_batch(ego_indices,
                                                        path_indices=episode_path_indices)

            wm.filter_per_step(states, step)

            if config.no_sign:
                ref_start = agent.DIM_EGO + agent.DIM_OTHERS + agent.DIM_VALIDITY
                for w in wm.good_worlds:
                    states[w][ref_start] = abs(states[w][ref_start])

            for w in wm.good_worlds:
                agent.buffer.handle_new_experience((states[w], w, episode_path_indices[w]))
                builder.increment_step(w)

            positions = builder.get_ego_positions_batch(ego_indices)
            for i, w in enumerate(VIZ_WORLDS):
                if w in wm.bad_worlds or w in wm.reached_worlds:
                    continue
                viz_list[i].record_step(positions[w, 0], positions[w, 1])

            logger.debug(f'状态构建完成，开始选择动作')
            actions = agent.select_action(states)

            _diag_init_step(config, agent, builder, ego_indices, episode_path_indices,
                            states, actions, step)

            actions = extend_action_to_3d(actions)
            logger.debug(f'动作选择完成，开始环境交互，动作形状: {actions.shape}')
            env.step_dynamics(actions)

            if step % 5 == 0:
                abs_np = builder.sim.absolute_self_observation_tensor().to_torch().cpu().numpy()
                for w in range(config.num_worlds):
                    a = ego_indices[w]
                    x, y = float(abs_np[w, a, 0]), float(abs_np[w, a, 1])
                    if abs(x) > 1e5 or abs(y) > 1e5:
                        logger.warning(f'[TELEPORT] world_{w} step={step} x={x:.1f} y={y:.1f}')

            logger.debug(f'环境交互完成，开始更新智能体')
            # PDMS 数据采集（每步一次，批量拉 tensor）
            info_np = env.sim.info_tensor().to_torch().cpu().numpy()
            abs_for_pdms = env.sim.absolute_self_observation_tensor().to_torch().cpu().numpy()
            partner_for_pdms = env.sim.partner_observations_tensor().to_torch().cpu().numpy()
            rel_np = env.sim.self_observation_tensor().to_torch().cpu().numpy()
            for w in wm.good_worlds:
                a = ego_indices[w]
                ego_x, ego_y = float(abs_for_pdms[w, a, 0]), float(abs_for_pdms[w, a, 1])
                ego_heading = float(abs_for_pdms[w, a, 7])
                ego_vel = float(rel_np[w, a, 0])
                partners = partner_for_pdms[w, a]
                off_road = float(info_np[w, a, 0]) > 0
                collision = (float(info_np[w, a, 1]) + float(info_np[w, a, 2])) > 0
                pid = episode_path_indices[w]
                t = builder.step_counter[w]
                path = builder.candidate_paths[w][a][pid]
                road_dist_ref = float(path['road_dist'][min(t, len(path['road_dist']) - 1)])
                scorers[w].update_step(
                    ego_pos=(ego_x, ego_y),
                    ego_vel=ego_vel,
                    ego_heading=ego_heading,
                    partners=partners,
                    off_road=off_road,
                    collision=collision,
                    delta_phi=float(states[w][ref_start_agent + 1]),
                    temporal_idx=t,
                    max_step=max_step,
                    road_dist_ref=road_dist_ref,
                    lat=float(states[w][ref_start_agent + 3]),
                    dt=config.dt,
                )

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

        # PDMS epoch logging (仅 surviving worlds，排除坐标系异常的 crash 场景)
        pdms_scores = [scorers[w].compute()
                       for w in range(config.num_worlds)
                       if w not in wm.bad_worlds and scorers[w].steps > 0]
        if pdms_scores:
            avg_score = np.mean([ps['driving_score'] for ps in pdms_scores])
            avg_comp = np.mean([ps['route_completion'] for ps in pdms_scores])
            total_coll = sum(ps['counts']['collision_steps'] for ps in pdms_scores)
            total_off = sum(ps['counts']['off_road_steps'] for ps in pdms_scores)
            logger.info(f'[PDMS] score={avg_score:.1f} completion={avg_comp:.1%} '
                        f'collisions={total_coll} off_road={total_off} '
                        f'(surviving: {len(pdms_scores)}/{config.num_worlds})')
            agent.epoch_history_pdms.append(pdms_scores)

        logger.info(f'开始保存轨迹图像')
        for viz in viz_list:
            if len(viz.actual_x) > 0:
                viz.save_plot(viz_dir, epoch + 1)

        save_info = {'env_name': 'examples'}
        agent.save(save_info=save_info)

        if wm.should_resample():
            logger.warning(
                f'[RESAMPLE] bad={len(wm.bad_worlds)}/{config.num_worlds}, '
                f'good<{config.num_worlds - config.max_bad_worlds}, triggering resample...'
            )
            ego_indices = wm.resample()

    logger.debug('训练完成')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='IDC-Bridge 训练脚本。所有默认值从 YAML 配置文件读取，CLI 参数可覆盖。')
    parser.add_argument('--config', type=str, default='/workspace/idc/src/configs/train.yaml',
                        help='YAML 配置文件路径')
    parser.add_argument('--data-dir', type=str, default='/workspace/wayo_data/data_json/training/',
                        help='Waymo tfrecord 数据目录')
    parser.add_argument('--num-worlds', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--init-penalty', type=float, default=None,
                        help='初始 ρ，0=纯追踪模式')
    parser.add_argument('--max-penalty', type=float, default=None)
    parser.add_argument('--amplifier-c', type=float, default=None)
    parser.add_argument('--lr-actor', type=float, default=None)
    parser.add_argument('--lr-critic', type=float, default=None)
    parser.add_argument('--dataset-size', type=int, default=None)
    parser.add_argument('--max-bad-worlds', type=int, default=None)
    parser.add_argument('--min-partner-density', type=float, default=None)
    parser.add_argument('--max-partner-density', type=int, default=None,
                        help='密度上限，min=0+max=30 表示全量')
    parser.add_argument('--dense-sample-size', type=int, default=None)
    parser.add_argument('--density-cache-file', type=str, default=None,
                        help='离线扫描的全量密度 JSON 路径')
    parser.add_argument('--file-dir', type=str, default=None)
    parser.add_argument('--load-model', dest='load_model', action='store_true', default=None)
    parser.add_argument('--no-load', dest='load_model', action='store_false')
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--save-freq', type=int, default=None)
    parser.add_argument('--fix-speed', action='store_true', default=None)
    parser.add_argument('--fix-heading', action='store_true', default=None)
    parser.add_argument('--no-sign', action='store_true', default=None)
    parser.add_argument('--no-road-penalty', action='store_true', default=None)
    parser.add_argument('--no-veh-penalty', action='store_true', default=None)
    args = parser.parse_args()

    # 加载 YAML 配置并用 CLI 覆盖
    cli_overrides = {
        'data_dir': args.data_dir,
        'num_worlds': args.num_worlds,
        'epochs': args.epochs,
        'seed': args.seed,
        'device': args.device,
        'init_penalty': args.init_penalty,
        'max_penalty': args.max_penalty,
        'amplifier_c': args.amplifier_c,
        'lr_actor': args.lr_actor,
        'lr_critic': args.lr_critic,
        'dataset_size': args.dataset_size,
        'max_bad_worlds': args.max_bad_worlds,
        'min_partner_density': args.min_partner_density,
        'max_partner_density': args.max_partner_density,
        'dense_sample_size': args.dense_sample_size,
        'density_cache_file': args.density_cache_file,
        'file_dir': args.file_dir,
        'load_model': args.load_model,
        'model_path': args.model_path,
        'save_freq': args.save_freq,
        'fix_speed': args.fix_speed,
        'fix_heading': args.fix_heading,
        'no_sign': args.no_sign,
        'no_road_penalty': args.no_road_penalty,
        'no_veh_penalty': args.no_veh_penalty,
    }
    config = build_config(args.config, cli_overrides)
    train(config)
