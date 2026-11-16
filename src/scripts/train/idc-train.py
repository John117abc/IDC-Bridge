import sys
import os
import glob
import random

import numpy as np
import torch

from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.dataset import SceneDataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from env.env_utils import extend_action_to_3d, get_env_config, load_scenes, get_ego_indices
from env.world_manager import WorldManager
from env.idc_state_builder import GPUDriveObservationBuilder
from agents.idc_agent import DiscreteIDCAgent
from utils import get_logger, VisualRecorder, TrajectoryVisualizer
from utils.config import build_config
from metrics import PDMSScorer

logger = get_logger('idc-train')


def _eps_at(epoch, config):
    """ε 调度：warmup 期纯随机（让 actor 学会跟踪所有候选），之后线性降到 eps_end。"""
    e0 = float(getattr(config, 'eps_start', 1.0))
    e1 = float(getattr(config, 'eps_end', 0.5))
    warm = int(getattr(config, 'warmup_epochs', 100))
    decay = max(int(getattr(config, 'eps_decay_epochs', 200)), 1)
    if epoch < warm:
        return e0
    if epoch < warm + decay:
        t = (epoch - warm) / decay
        return e0 + t * (e1 - e0)
    return e1


def _maybe_install_candidates(builder, ego_indices, env, config, logger, n=None):
    """若启用多候选：生成并注入 builder；返回实际生效的 K。

    n=None 时取 config.num_candidates。地图来源即使 n=1 也要安装
    （阶段 1 的"单条地图参考"就是这么来的，不能用专家路径代替）。
    """
    num = int(n if n is not None else (getattr(config, 'num_candidates', 1) or 1))
    source = str(getattr(config, 'cand_source', 'map')).lower()
    if source != 'map' and num <= 1:
        return 1
    try:
        from env.candidate_sampler import SamplerConfig, install_candidates
        sc = SamplerConfig(
            lat_offsets=tuple(getattr(config, 'cand_lat_offsets',
                                      (-3.5, -1.6, -0.8, 0.0, 0.8, 1.6, 3.5))),
            allow_wide=bool(getattr(config, 'cand_allow_wide', False)),
            dt_shifts=tuple(getattr(config, 'cand_dt_shifts', (0.0,))),
            k_select=int(getattr(config, 'cand_k_select', 4)),
            delta=float(getattr(config, 'cand_delta', 0.8)),
            d_safe=float(getattr(config, 'cand_d_safe', 0.8)),
            corridor_bad_frac=float(getattr(config, 'cand_bad_frac', 0.05)),
            clear_abs_min=float(getattr(config, 'cand_clear_min', 0.15)),
        )
        maps = env.sim.map_observation_tensor().to_torch().cpu().numpy()
        source = str(getattr(config, 'cand_source', 'map')).lower()
        if source == 'map':
            # 无特权信息：候选只由 当前位姿 + 地图 生成
            from env.map_candidates import install_map_candidates
            abs_np = env.sim.absolute_self_observation_tensor(
            ).to_torch().cpu().numpy()
            rel_np = env.sim.self_observation_tensor(
            ).to_torch().cpu().numpy()
            W = builder.num_worlds
            poses = [(float(abs_np[w, ego_indices[w], 0]),
                      float(abs_np[w, ego_indices[w], 1]),
                      float(abs_np[w, ego_indices[w], 7])) for w in range(W)]
            speeds = [float(np.clip(rel_np[w, ego_indices[w], 2], 1.0, 20.0))
                      for w in range(W)]
            K, per_world = install_map_candidates(
                builder, ego_indices, maps, poses, speeds, sc,
                k_select=max(1, num))
            # ---- 场景筛选：起步就急转（专家净位移方向与 ego 初始朝向差 >60°）
            # 的场景，地图参考无法确定该走哪条分支（占 ~10%）。
            # 只用于"选场景"，不作为模型输入；这些 world 标记为坏，
            # 让世界管理器在下次 resample 时替换掉。
            amb = []
            ex_all = builder.expert_pos.detach().cpu().numpy()
            for w in range(builder.num_worlds):
                a = int(ego_indices[w])
                ex = ex_all[w, a]
                ex = ex[np.all(np.abs(ex) < 1e4, axis=1)]
                if len(ex) < 5:
                    continue
                d = ex[-1] - ex[0]
                if np.linalg.norm(d) < 3.0:
                    continue
                net = float(np.degrees(np.arctan2(d[1], d[0])))
                turn = abs((net - np.degrees(poses[w][2]) + 180) % 360 - 180)
                if turn > 60.0:
                    amb.append(w)
            builder._ambiguous_worlds = amb
            ks = [len(v) for v in per_world.values()]
            logger.info(f'[CANDS] 地图候选注入: K={K} '
                        f'每world中位={np.median(ks):.0f} '
                        f'车速中位={np.median(speeds):.1f} m/s '
                        f'起步歧义world={len(amb)}')
            return K
        if num <= 1:
            return 1
        K = install_candidates(builder, ego_indices, maps, sc)
        logger.info(f'[CANDS] 专家候选注入: K={K} '
                    f'(offsets={sc.lat_offsets}, k_select={sc.k_select})')
        return K
    except Exception as e:                      # noqa: BLE001
        logger.warning(f'[CANDS] 注入失败，退回单路径: {e}')
        try:
            builder.generate_candidate_paths(ego_indices, num_paths=1,
                                             use_expert_path=True)
        except Exception:
            pass
        return 1


def _diag_init_step(config, agent, builder, ego_indices, episode_path_indices,
                    states, actions, step):
    if step != 0:
        return

    # 用窗口形式传给 Transformer Actor
    batch_windows = []
    for i in [0, 1]:
        win = np.tile(states[i], (config.window_size, 1)).astype(np.float32)
        batch_windows.append(win)
    window_tensor = torch.from_numpy(np.stack(batch_windows)).to(agent.device)
    with torch.no_grad():
        norm_action_raw = agent._policy(window_tensor)
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
            f'raw_steer={rsteer:.3f} raw_acc={racc:.3f} ref_spd={ref_spd0:.2f} '
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
    builder.generate_candidate_paths(ego_indices, num_paths=1,
                                       use_expert_path=getattr(config, 'use_expert_path', True))
    K_PATHS = _maybe_install_candidates(builder, ego_indices, env, config, logger)

    agent = DiscreteIDCAgent(env, config, config.device, builder, ego_indices)

    wm = WorldManager(env, builder, agent, all_files, config, logger)
    for _w in getattr(builder, '_ambiguous_worlds', []):
        wm.bad_worlds.add(_w)          # 起步歧义场景：交给世界管理器替换
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

    # ε 调度锚点：从"本次训练的起点"算起，而不是绝对 epoch。
    # 否则续训（如从 epoch=2400 的基线热启动）会直接落在衰减后的阶段，
    # 跳过 ε=1.0 的 warmup —— 而那正是"让 actor 学会跟踪所有候选"的关键。
    _cfg_anchor = int(getattr(config, 'eps_anchor_epoch', -1))
    EPS_ANCHOR = (_cfg_anchor if _cfg_anchor >= 0
                  else int(getattr(agent, 'globe_eps', 0) or 0))
    logger.info(f'[CANDS] eps 调度锚点 = epoch {EPS_ANCHOR}')

    for epoch in range(epochs):
        epoch += current_epoch
        obs = env.reset()
        for w in range(config.num_worlds):
            builder.reset_world_step(w, 0)
            agent.reset_world_state(w)
        builder.clear_cache()
        wm.begin_epoch()

        # 检测 epoch 开头已 done 的世界（轨迹数据异常/已到终点）
        done_np = env.get_dones().cpu().numpy()
        for w in range(config.num_worlds):
            if w in wm.bad_worlds:
                continue
            a = ego_indices[w]
            if done_np[w, a] > 0.5:
                wm.reached_worlds.add(w)

        epoch_history = []

        # 轨迹图：只对少量 world 记录，且按间隔落盘（原实现每个 epoch × 10 个 world，
        # 1500 epoch 会产出 1.5 万张图）
        n_viz = max(int(getattr(config, 'traj_plot_max_worlds', 4)), 0)
        VIZ_WORLDS = wm.good_worlds[:n_viz] if n_viz > 0 else []
        viz_list = [TrajectoryVisualizer(builder, w, ego_indices[w])
                    for w in VIZ_WORLDS]

        # ε-greedy 选候选（整条 episode 固定一条）：
        #   warmup 期 ε=1 纯随机 → 逼 actor 学会跟踪所有横向候选；
        #   之后 ε 线性下降，非探索分支暂时固定为中心候选（等 M1 的 J/V 选路）。
        # ---- 课程：前 curriculum_epochs 轮只装中心候选（先学会跟一条地图路径），
        #      之后换成多候选 + ε 探索（学会跟横向偏移）。全程只用位姿+地图。
        _curr = int(getattr(config, 'curriculum_epochs', 0) or 0)
        _n_target = 1 if (_curr > 0 and epoch < _curr) else int(
            getattr(config, 'num_candidates', 5))
        if _n_target != K_PATHS:
            K_PATHS = _maybe_install_candidates(builder, ego_indices, env,
                                                config, logger, n=_n_target)
            logger.info(f'[CANDS] 课程切换 epoch={epoch} → K={K_PATHS}')

        eps = _eps_at(epoch - EPS_ANCHOR, config)
        if K_PATHS > 1:
            if np.random.rand() < eps:
                episode_path_indices = [int(np.random.randint(0, K_PATHS))
                                        for _ in range(config.num_worlds)]
            else:
                episode_path_indices = [0 for _ in range(config.num_worlds)]
        else:
            episode_path_indices = [0 for _ in range(config.num_worlds)]
        if K_PATHS > 1 and (epoch % 20 == 0):
            hist = np.bincount(episode_path_indices, minlength=K_PATHS)
            logger.info(f'[CANDS] epoch={epoch+1} eps={eps:.2f} '
                        f'选择分布={hist.tolist()} K={K_PATHS}')

        scorers = [PDMSScorer(config) for _ in range(config.num_worlds)]
        ref_start_agent = agent.DIM_EGO + agent.DIM_OTHERS + agent.DIM_VALIDITY
        # 投影进度索引在 state 里的位置（temporal 维）
        temp_idx_pos = ref_start_agent + agent.DIM_REF_ERROR

        for step in range(max_step):
            if step % 10 == 0:
                logger.info(f'回合 {epoch+1}/{config.epochs}, 步数 {step+1}/{max_step}')
            else:
                logger.debug(f'回合 {epoch+1}/{config.epochs}, 步数 {step+1}/{max_step}')

            # 批量拉取 tensor（一次 GPU→CPU），后续 state building + positions 复用
            abs_np_step = env.sim.absolute_self_observation_tensor().to_torch().cpu().numpy()
            rel_np_step = env.sim.self_observation_tensor().to_torch().cpu().numpy()
            partner_np_step = env.sim.partner_observations_tensor().to_torch().cpu().numpy()
            states = builder.get_idc_observations_batch(
                ego_indices, path_indices=episode_path_indices,
                _abs_np=abs_np_step, _rel_np=rel_np_step, _partner_np=partner_np_step)

            wm.filter_per_step(states, step)

            ref_start = agent.DIM_EGO + agent.DIM_OTHERS + agent.DIM_VALIDITY
            for w in wm.good_worlds:
                if config.no_sign:
                    states[w][ref_start] = abs(states[w][ref_start])
                priority = abs(states[w][ref_start]) + abs(states[w][ref_start + 1]) * 5.0 + 1e-6

                if w not in agent.state_windows:
                    agent.state_windows[w] = agent._deque(maxlen=config.window_size)
                agent.state_windows[w].append(states[w])
                win_list = list(agent.state_windows[w])
                window = np.tile(win_list[0], (config.window_size, 1)).astype(np.float32)
                window[-len(win_list):] = np.array(win_list)

                agent.buffer.handle_new_experience(
                    (window, w, episode_path_indices[w]), priority=priority)
                builder.increment_step(w)

            for w in list(wm.bad_worlds) + list(wm.reached_worlds):
                agent.reset_world_state(w)

            positions = builder.get_ego_positions_batch(ego_indices, _abs_np=abs_np_step)
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
                # 用"投影进度索引"而不是步数时钟：否则 completion 还是时间进度
                t = int(round(float(states[w][temp_idx_pos])))
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

            if agent.buffer.should_start_training():
                logger.debug(f'开始训练: global_step={agent.global_step}, buffer size={len(agent.buffer)}')
                critic_loss, actor_loss = agent.update()
                logger.debug(f'训练完成: global_step={agent.global_step}')

                if critic_loss is not None or actor_loss is not None:
                    # None = 本步没轮到 PEV/PIM（pev_interval / pim_interval），不是 NaN
                    _cl = '   -   ' if critic_loss is None else f'{critic_loss:9.4f}'
                    _al = '   -   ' if actor_loss is None else f'{actor_loss:9.4f}'
                    logger.info(f'Critic Loss: {_cl}, Actor Loss: {_al}, '
                                f'Penalty Coefficient (rho): {agent.rho:.4f}')
                    if actor_loss is not None:
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

        plot_interval = max(int(getattr(config, 'traj_plot_interval', 20)), 1)
        if viz_list and ((epoch + 1) % plot_interval == 0 or (epoch + 1) == 1):
            logger.info(f'保存轨迹图像 (epoch={epoch + 1}, 每 {plot_interval} 轮)')
            # 绘图是否画专家轨迹：与 use_expert_path（路径生成方式）解耦，
            # 现在参考是地图生成的，图里画专家只会造成混淆
            use_expert = bool(getattr(config, 'plot_show_expert', False))
            for viz in viz_list:
                if len(viz.actual_x) > 0:
                    viz.save_plot(viz_dir, epoch + 1, use_expert_path=use_expert)
            try:
                from utils.traj_visualizer import make_montage
                _m = make_montage(viz_dir, epoch + 1)
                if _m:
                    logger.info(f'[VIZ] 蒙太奇 {_m}')
            except Exception as _e:            # noqa: BLE001
                logger.debug(f'[VIZ] montage failed: {_e}')

        save_info = {'env_name': 'examples'}
        agent.save(save_info=save_info)

        if wm.should_resample():
            logger.warning(
                f'[RESAMPLE] bad={len(wm.bad_worlds)}/{config.num_worlds}, '
                f'good<{config.num_worlds - config.max_bad_worlds}, triggering resample...'
            )
            ego_indices = wm.resample()
            if K_PATHS > 1:
                K_PATHS = _maybe_install_candidates(builder, ego_indices, env,
                                                    config, logger)
            for _w in getattr(builder, '_ambiguous_worlds', []):
                wm.bad_worlds.add(_w)
            for w in range(config.num_worlds):
                agent.reset_world_state(w)

    logger.debug('训练完成')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='IDC-Bridge 训练脚本。所有默认值从 YAML 配置文件读取，CLI 参数可覆盖。')
    # 默认配置路径按脚本位置推导（原来硬编码 /workspace/idc/src/configs/train.yaml，
    # 换目录即失效）
    _default_cfg = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', 'configs', 'train.yaml'))
    parser.add_argument('--config', type=str, default=_default_cfg,
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
    parser.add_argument('--resample-interval', type=int, default=None,
                        help='每 N 个 epoch 触发一次滚动替换（0=关闭）')
    parser.add_argument('--resample-ratio', type=float, default=None,
                        help='每次替换的 world 比例（0~1，默认 0.4）')
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
    # ---- 表示层消融开关 ----
    parser.add_argument('--state-mode', type=str, default=None,
                        choices=['ego', 'abs'], help='ego=自车系(默认) / abs=绝对坐标')
    parser.add_argument('--ref-mode', type=str, default=None,
                        choices=['nearest', 'index'],
                        help='nearest=按位置投影(默认) / index=逐帧+1')
    parser.add_argument('--obs-norm', type=str, default=None,
                        help='true/false，输入在线归一化')
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
        'resample_interval': args.resample_interval,
        'resample_ratio': args.resample_ratio,
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
        'state_mode': args.state_mode,
        'ref_mode': args.ref_mode,
        'obs_norm': (None if args.obs_norm is None
                     else args.obs_norm.lower() == 'true'),
    }
    config = build_config(args.config, cli_overrides)
    train(config)
