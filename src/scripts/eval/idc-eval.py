import sys
import os

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

logger = get_logger('idc-agent')


def evaluate(args):
    print("Data root:", args.data_dir)

    data_loader = SceneDataLoader(
        root=args.data_dir,
        batch_size=args.num_worlds,
        dataset_size=args.num_worlds,
        sample_with_replacement=True,
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

    recorder = VisualRecorder(
        num_worlds=args.num_worlds,
        save_dir="/workspace/data/gifs",
        fps=5,
    )

    scenes = load_scenes(env)
    builder = GPUDriveObservationBuilder(env, scenes)
    max_step = builder.EXPERT_TRAJ_LEN

    ego_indices = get_ego_indices(env, args.num_worlds)
    builder.generate_candidate_paths(ego_indices, num_paths=3)

    agent = DiscreteIDCAgent(env, args, args.device, builder, ego_indices)

    # eval 不需要密度扫描，只加载已有缓存
    all_files = []
    wm = WorldManager(env, builder, agent, all_files, args, logger, compute_density=False)
    wm.filter_initial(ego_indices)

    logger.info(f'正在加载模型: {args.model_path}')
    agent.load(args.model_path)
    agent.rho = args.init_penalty
    agent.max_penalty = args.max_penalty
    logger.info(f'评估模式: rho={agent.rho}, max_penalty={agent.max_penalty}')

    logger.info('正在生成损失曲线图...')
    LossPlotter(agent.history_loss, f'{args.file_dir}/loss_img', 'idc-waymo-v1.0').plot_all()

    logger.info(f'评估开始: epochs={args.epochs}, num_worlds={args.num_worlds}, max_steps={max_step}')

    viz_dir = os.path.join(args.file_dir, 'eval_plots')
    os.makedirs(viz_dir, exist_ok=True)

    for epoch in range(args.epochs):
        obs = env.reset()
        for w in range(args.num_worlds):
            builder.reset_world_step(w, 0)
        builder.clear_cache()

        viz_list = [TrajectoryVisualizer(builder, w, ego_indices[w])
                    for w in wm.good_worlds]
        VIZ_WORLDS = list(wm.good_worlds)

        for step in range(max_step):
            logger.info(f'回合 {epoch+1}/{args.epochs}, 步数 {step+1}/{max_step}')

            path_indices = [0 for _ in range(args.num_worlds)]
            states = builder.get_idc_observations_batch(ego_indices, path_indices=path_indices)

            wm.filter_per_step(states, step)

            for w in range(args.num_worlds):
                builder.increment_step(w)

            positions = builder.get_ego_positions_batch(ego_indices)
            for i, w in enumerate(VIZ_WORLDS):
                if w in wm.bad_worlds:
                    continue
                if i < len(viz_list):
                    viz_list[i].record_step(positions[w, 0], positions[w, 1])

            actions = agent.select_action(states, deterministic=True)
            actions = extend_action_to_3d(actions)
            logger.debug(f'动作选择完成，开始环境交互，动作形状: {actions.shape}')
            env.step_dynamics(actions)

            good_envs = [w for w in range(args.num_worlds) if w not in wm.bad_worlds]
            if step % recorder.fps == 0 and good_envs:
                imgs = env.vis.plot_simulator_state(
                    env_indices=good_envs,
                    time_steps=[epoch] * len(good_envs),
                    zoom_radius=70,
                )
                for i, w in enumerate(good_envs):
                    recorder.frames[f"env_{w}"].append(img_from_fig(imgs[i]))

        for viz in viz_list:
            if len(viz.actual_x) > 0:
                viz.save_plot(viz_dir, epoch + 1)

    recorder.save_all_gifs()
    logger.debug('评估完成')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--num-worlds', type=int, default=10)
    parser.add_argument('--max-agents', type=int, default=1)
    parser.add_argument('--max-steps', type=int, default=90)
    parser.add_argument('--epochs', type=int, default=1)
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
    parser.add_argument('--max-bad-worlds', type=int, default=100)
    parser.add_argument('--min-partner-density', type=float, default=2.0)
    parser.add_argument('--dense-sample-size', type=int, default=500)
    parser.add_argument('--seed', type=int, default=20)
    parser.add_argument('--save-freq', type=int, default=1)
    parser.add_argument('--file-dir', type=str, default="/workspace/data")
    parser.add_argument('--load-model', type=bool, default=True)
    parser.add_argument('--model-path', type=str,
                        default="/workspace/data/checkpoints/20260525/idc-waymo-v1.0_examples_135820_episode=155.pth")
    args = parser.parse_args()
    evaluate(args)
