import sys
import os
import random

import numpy as np
import torch

from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.visualize.utils import img_from_fig

sys.path.insert(0, '/workspace/idc/src')
from env.env_utils import extend_action_to_3d, get_env_config, load_scenes, get_ego_indices
from env.world_manager import WorldManager
from env.idc_state_builder import GPUDriveObservationBuilder
from agents.idc_agent import DiscreteIDCAgent
from utils import get_logger, VisualRecorder, TrajectoryVisualizer, LossPlotter
from utils.config import build_config
from metrics import PDMSScorer, RolloutPDMSScorer
from metrics.plotter import print_pdms_table, plot_pdms_radar, plot_pdms_bar

logger = get_logger('idc-eval')


def _record_gif_frame(env, recorder, config, epoch, step, ego_indices, wm, selected_worlds):
    """根据 gif_view_mode 录制一帧到 recorder"""
    view_mode = config.gif_view_mode
    zoom = config.gif_zoom_radius

    if view_mode == "agent_pov":
        for w in selected_worlds:
            if w in wm.bad_worlds or w in wm.reached_worlds:
                continue
            result = env.vis.plot_agent_observation(
                agent_idx=ego_indices[w], env_idx=w)
            fig = result[0] if isinstance(result, tuple) else result
            if fig is not None:
                recorder.frames[f"env_{w}"].append(img_from_fig(fig))
    else:
        if view_mode == "bird_3d":
            env.vis.render_3d = True
        imgs = env.vis.plot_simulator_state(
            env_indices=selected_worlds,
            time_steps=[epoch] * len(selected_worlds),
            zoom_radius=zoom,
        )
        if view_mode == "bird_3d":
            env.vis.render_3d = False
        for i, w in enumerate(selected_worlds):
            recorder.frames[f"env_{w}"].append(img_from_fig(imgs[i]))


def _select_gif_worlds(config, good_envs):
    """根据 gif_world_selection + gif_max_worlds 选出要录制的 world 列表"""
    max_n = config.gif_max_worlds
    if max_n == 0 or max_n >= len(good_envs):
        return good_envs

    if config.gif_world_selection == "random":
        return random.sample(good_envs, max_n)
    # "first": 前 N 个
    return good_envs[:max_n]


def evaluate(config):
    print("Data root:", config.data_dir)

    data_loader = SceneDataLoader(
        root=config.data_dir,
        batch_size=config.num_worlds,
        dataset_size=config.num_worlds,
        sample_with_replacement=True,
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

    gif_enabled = getattr(config, 'gif_enabled', True)
    recorder = VisualRecorder(
        num_worlds=config.num_worlds,
        save_dir=config.gif_save_dir,
        fps=config.gif_record_interval,
    )

    scenes = load_scenes(env)
    builder = GPUDriveObservationBuilder(env, scenes)
    max_step = builder.EXPERT_TRAJ_LEN

    ego_indices = get_ego_indices(env, config.num_worlds)
    builder.generate_candidate_paths(ego_indices, num_paths=1)

    agent = DiscreteIDCAgent(env, config, config.device, builder, ego_indices)

    all_files = []
    wm = WorldManager(env, builder, agent, all_files, config, logger, compute_density=False)
    wm.filter_initial(ego_indices)

    logger.info(f'正在加载模型: {config.model_path}')
    agent.load(config.model_path)
    agent.rho = config.init_penalty
    agent.max_penalty = config.max_penalty
    logger.info(f'评估模式: rho={agent.rho}, max_penalty={agent.max_penalty}')

    logger.info('正在生成损失曲线图...')
    LossPlotter(agent.history_loss, f'{config.file_dir}/loss_img', 'idc-waymo-v1.0').plot_all()

    logger.info(f'评估开始: epochs={config.epochs}, num_worlds={config.num_worlds}, max_steps={max_step}')
    if gif_enabled:
        logger.info(f'GIF: view={config.gif_view_mode} zoom={config.gif_zoom_radius} '
                    f'playback={config.gif_fps}fps interval={config.gif_record_interval} '
                    f'max_worlds={config.gif_max_worlds}')
    else:
        logger.info('GIF: disabled')

    viz_dir = os.path.join(config.file_dir, 'eval_plots')
    os.makedirs(viz_dir, exist_ok=True)

    for epoch in range(config.epochs):
        obs = env.reset()
        for w in range(config.num_worlds):
            builder.reset_world_step(w, 0)
            agent.reset_world_state(w)
        builder.clear_cache()
        wm.reached_worlds.clear()

        done_np = env.get_dones().cpu().numpy()
        for w in range(config.num_worlds):
            if w in wm.bad_worlds:
                continue
            a = ego_indices[w]
            if done_np[w, a] > 0.5:
                wm.reached_worlds.add(w)

        viz_list = [TrajectoryVisualizer(builder, w, ego_indices[w])
                    for w in wm.good_worlds]
        VIZ_WORLDS = list(wm.good_worlds)

        gif_worlds = _select_gif_worlds(config, VIZ_WORLDS)

        scorers = [PDMSScorer(config) for _ in range(config.num_worlds)]
        ref_start_agent = agent.DIM_EGO + agent.DIM_OTHERS + agent.DIM_VALIDITY

        for step in range(max_step):
            logger.info(f'回合 {epoch+1}/{config.epochs}, 步数 {step+1}/{max_step}')

            path_indices = [0 for _ in range(config.num_worlds)]
            states = builder.get_idc_observations_batch(ego_indices, path_indices=path_indices)

            # Rollout PDMS — 仅在 step 0 对前 3 个 good world 做一次前向推演
            if step == 0 and epoch == 0:
                rollout_scores = {}
                roll_scorer = RolloutPDMSScorer(agent, config, path_idx=0)
                for w in wm.good_worlds[:3]:
                    try:
                        roll = roll_scorer.compute_rollout_pdms(states[w], w, ego_indices[w])
                        rollout_scores[w] = roll
                        logger.info(f'[PDMS-rollout] world_{w} predicted_score={roll["driving_score"]:.1f}')
                    except Exception as e:
                        logger.debug(f'[PDMS-rollout] world_{w} failed: {e}')

            wm.filter_per_step(states, step)

            for w in list(wm.bad_worlds) + list(wm.reached_worlds):
                agent.reset_world_state(w)

            for w in wm.good_worlds:
                builder.increment_step(w)

            positions = builder.get_ego_positions_batch(ego_indices)
            for i, w in enumerate(VIZ_WORLDS):
                if w in wm.bad_worlds or w in wm.reached_worlds:
                    continue
                if i < len(viz_list):
                    viz_list[i].record_step(positions[w, 0], positions[w, 1])

            actions = agent.select_action(states, deterministic=True)
            actions = extend_action_to_3d(actions)
            logger.debug(f'动作选择完成，开始环境交互，动作形状: {actions.shape}')
            env.step_dynamics(actions)

            # PDMS 数据采集
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
                pid = 0
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
                    max_step=len(path['pos']),
                    road_dist_ref=road_dist_ref,
                    lat=float(states[w][ref_start_agent + 3]),
                    dt=config.dt,
                )

            if gif_enabled and step % recorder.fps == 0 and gif_worlds:
                _record_gif_frame(env, recorder, config, epoch, step,
                                  ego_indices, wm, gif_worlds)

        for viz in viz_list:
            if len(viz.actual_x) > 0:
                viz.save_plot(viz_dir, epoch + 1)

        # PDMS epoch 报告（仅 surviving worlds）
        pdms_scores = [scorers[w].compute()
                       for w in range(config.num_worlds)
                       if w not in wm.bad_worlds and scorers[w].steps > 0]
        if pdms_scores:
            print_pdms_table(pdms_scores, logger,
                             total_worlds=config.num_worlds)

            pdms_plot_dir = os.path.join(config.file_dir, 'pdms_plots')
            os.makedirs(pdms_plot_dir, exist_ok=True)

            plot_pdms_radar(pdms_scores,
                            os.path.join(pdms_plot_dir, f'pdms_radar_epoch{epoch+1}.png'),
                            title=f'PDMS Epoch {epoch+1}')

            bar_data = [{'world_idx': w, **scorers[w].compute()}
                        for w in range(config.num_worlds)
                        if w not in wm.bad_worlds and scorers[w].steps > 0]
            plot_pdms_bar(bar_data[:20],
                         os.path.join(pdms_plot_dir, f'pdms_bar_epoch{epoch+1}.png'),
                         title=f'Per-World Driving Score Epoch {epoch+1}')

    if gif_enabled:
        recorder.save_all_gifs(custom_fps=config.gif_fps)
    logger.debug('评估完成')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='IDC-Bridge 评估脚本。所有默认值从 YAML 配置文件读取，CLI 参数可覆盖。')
    parser.add_argument('--config', type=str, default='configs/eval.yaml',
                        help='YAML 配置文件路径')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Waymo tfrecord 数据目录')
    parser.add_argument('--model-path', type=str, required=True,
                        help='模型 checkpoint 路径')
    parser.add_argument('--num-worlds', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--init-penalty', type=float, default=None)
    parser.add_argument('--max-penalty', type=float, default=None)
    parser.add_argument('--file-dir', type=str, default=None)
    parser.add_argument('--gif-enabled', type=str, default=None,
                        help='是否生成 GIF (true/false)')
    parser.add_argument('--gif-view-mode', type=str, default=None,
                        help='bird_2d / bird_3d / agent_pov')
    parser.add_argument('--gif-max-worlds', type=int, default=None,
                        help='最多录制 world 数 (0=全部)')
    parser.add_argument('--gif-zoom-radius', type=int, default=None)
    parser.add_argument('--gif-fps', type=int, default=None,
                        help='GIF 播放速度（帧/秒）')
    parser.add_argument('--gif-record-interval', type=int, default=None,
                        help='每隔 N 个 env step 录一帧')
    parser.add_argument('--gif-world-selection', type=str, default=None,
                        help='first / random')
    parser.add_argument('--min-partner-density', type=float, default=None)
    parser.add_argument('--max-partner-density', type=int, default=None)
    parser.add_argument('--dense-sample-size', type=int, default=None)
    parser.add_argument('--density-cache-file', type=str, default=None)
    parser.add_argument('--no-road-penalty', action='store_true', default=None)
    parser.add_argument('--no-veh-penalty', action='store_true', default=None)

    args = parser.parse_args()

    cli_overrides = {
        'data_dir': args.data_dir,
        'model_path': args.model_path,
        'num_worlds': args.num_worlds,
        'epochs': args.epochs,
        'seed': args.seed,
        'device': args.device,
        'init_penalty': args.init_penalty,
        'max_penalty': args.max_penalty,
        'file_dir': args.file_dir,
        'gif_enabled': args.gif_enabled,
        'gif_view_mode': args.gif_view_mode,
        'gif_max_worlds': args.gif_max_worlds,
        'gif_zoom_radius': args.gif_zoom_radius,
        'gif_fps': args.gif_fps,
        'gif_record_interval': args.gif_record_interval,
        'gif_world_selection': args.gif_world_selection,
        'min_partner_density': args.min_partner_density,
        'max_partner_density': args.max_partner_density,
        'dense_sample_size': args.dense_sample_size,
        'density_cache_file': args.density_cache_file,
        'no_road_penalty': args.no_road_penalty,
        'no_veh_penalty': args.no_veh_penalty,
    }
    config = build_config(args.config, cli_overrides)

    if isinstance(config.gif_enabled, str):
        config.gif_enabled = config.gif_enabled.lower() == 'true'

    evaluate(config)
