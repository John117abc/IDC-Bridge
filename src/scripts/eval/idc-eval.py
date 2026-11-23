import sys
import os
import random

import numpy as np
import torch

from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.visualize.utils import img_from_fig

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from env.env_utils import extend_action_to_3d, get_env_config, load_scenes, get_ego_indices
from env.world_manager import WorldManager
from env.idc_state_builder import GPUDriveObservationBuilder
from agents.idc_agent import DiscreteIDCAgent
from utils import get_logger, VisualRecorder, TrajectoryVisualizer, LossPlotter
from utils.config import build_config
from metrics import PDMSScorer, RolloutPDMSScorer
from metrics.plotter import print_pdms_table, plot_pdms_radar, plot_pdms_bar

logger = get_logger('idc-eval')


def _render_storyboards(trace, env, builder, config, ego_indices):
    """用 storyboard 风格离线渲染 GIF + 关键时刻拼图。

    取代 gpudrive 自带的 plot_simulator_state：那个只画道路细线和车辆圆点，
    既无车头朝向也无尾迹，而且按 time_steps 取帧容易取错导致静止图。
    """
    import time

    import numpy as np

    from utils.storyboard import render_world

    out_dir = os.path.join(config.gif_save_dir, time.strftime('%Y%m%d'))
    maps = env.sim.map_observation_tensor().to_torch().cpu().numpy()
    margin = float(getattr(config, 'gif_zoom_radius', 70)) * 0.5
    for w, tw in trace.items():
        a = ego_indices[w]
        ep = builder.expert_pos[w, a]
        ep = ep.detach().cpu().numpy() if hasattr(ep, 'detach') else np.asarray(ep)
        ep = np.asarray(ep)[10:]        # 只画决策点（frame 10）之后的未来
        _cands = None
        try:
            _ai = int(a)
            if w in builder.candidate_paths and _ai in builder.candidate_paths[w]:
                _cands = [np.asarray((x['pos'].detach().cpu().numpy()
                                      if hasattr(x['pos'], 'detach') else x['pos']))
                          for x in builder.candidate_paths[w][_ai]]
        except Exception:
            _cands = None
        _seq = tw.get('cands') if isinstance(tw, dict) else None
        _sel = tw.get('sel') if isinstance(tw, dict) else None
        png, gif = render_world(tw, maps[w], ep, out_dir, w,
                                fps=int(getattr(config, 'gif_fps', 12)),
                                margin=margin, candidates=_cands,
                                cands_seq=_seq, sel_seq=_sel,
                                cands_pre=0 if getattr(config, 'planner_runtime', '')
                                else 11)
        if gif:
            logger.info(f'[storyboard] world_{w} -> {os.path.basename(gif)}'
                        + (f' + {os.path.basename(png)}' if png else ''))


def _dump_trace(trace, env, builder, config, ego_indices):
    """把选中的 world 的逐帧位姿 + 地图 + 专家路径落成 npz，供离线画图。"""
    import time

    import numpy as np

    date_dir = os.path.join(config.file_dir, 'traces', time.strftime('%Y%m%d'))
    os.makedirs(date_dir, exist_ok=True)

    out = {}
    for w, t in trace.items():
        out[f'ego_{w}'] = np.asarray(t['ego'], dtype=np.float32)
        out[f'info_{w}'] = np.asarray(t['info'], dtype=np.float32)
        out[f'partners_{w}'] = np.array(t['partners'], dtype=object)

    maps = env.sim.map_observation_tensor().to_torch().cpu().numpy()
    for w in trace:
        out[f'map_{w}'] = maps[w].astype(np.float32)
        a = ego_indices[w]
        ep = builder.expert_pos[w, a]
        ep = ep.detach().cpu().numpy() if hasattr(ep, 'detach') else np.asarray(ep)
        ep = np.asarray(ep)[10:]        # 只画决策点（frame 10）之后的未来
        out[f'expert_{w}'] = ep.astype(np.float32)

    path = os.path.join(date_dir, f'trace_{time.strftime("%H%M%S")}.npz')
    np.savez_compressed(path, **out)
    print(f'[TRACE] saved {path}  worlds={list(trace.keys())}')


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
            # 这里必须用"当前仿真步数"，不能用回合号 epoch：
            # plot_simulator_state 是按 time_steps 去取已记录轨迹的某一帧，
            # 传 epoch（eval 恒为 0）会导致每一帧都渲染第 0 帧，GIF 成了静止图。
            time_steps=[step] * len(selected_worlds),
            zoom_radius=zoom,
        )
        if view_mode == "bird_3d":
            env.vis.render_3d = False
        for i, w in enumerate(selected_worlds):
            recorder.frames[f"env_{w}"].append(img_from_fig(imgs[i]))


def _select_gif_worlds(config, good_envs):
    """根据 gif_worlds / gif_world_selection + gif_max_worlds 选出要录制的 world 列表"""
    _want = getattr(config, 'gif_worlds', '') or ''
    if _want:
        _ids = [int(x) for x in str(_want).split(',') if str(x).strip() != '']
        _sel = [w for w in _ids if w in good_envs]
        if _sel:
            return _sel
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
        sample_with_replacement=False,
        shuffle=True,
        seed=config.seed if config.seed is not None else random.randint(0, 999999),
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
    try:
        import json as _json
        _idx0 = list(env.data_loader.indices)[:config.num_worlds]
        _fs0 = [os.path.basename(str(env.data_loader.dataset[k])) for k in _idx0]
        with open('/workspace/data/world_files.json', 'w') as _fh:
            _json.dump(_fs0, _fh)
        logger.info(f'[EVAL] world files dumped: {len(_fs0)}')
    except Exception as _e:
        logger.info(f'[EVAL] world dump failed: {_e}')
    max_step = builder.EXPERT_TRAJ_LEN

    ego_indices = get_ego_indices(env, config.num_worlds)
    builder.generate_candidate_paths(ego_indices, num_paths=1,
                                       use_expert_path=getattr(config, 'use_expert_path', True))

    # A/B 对照：固定安装候选集，索引语义确定（CenterIndex = K//2）
    _cand_fixed = int(getattr(config, 'cand_fixed', 0) or 0)
    if _cand_fixed > 0:
        from env.candidate_sampler import install_fixed
        _offs = {3: (-0.8, 0.0, 0.8),
                 5: (-1.6, -0.8, 0.0, 0.8, 1.6)}.get(
                     _cand_fixed, (-0.8, 0.0, 0.8))
        _maps = env.sim.map_observation_tensor().to_torch().cpu().numpy()
        _K = install_fixed(builder, ego_indices, _maps, _offs)
        logger.info(f'[AB] 固定候选安装完成: K={_K} offsets={_offs} '
                    f'center_idx={_K // 2}')

    _ext_path = getattr(config, 'external_candidates', '') or ''
    if _ext_path:
        import numpy as _np
        from env.candidate_sampler import install_external
        _data = _np.load(_ext_path, allow_pickle=True)
        _names = [str(n) for n in _data['names']]
        _n2i = {n: i for i, n in enumerate(_names)}
        _idx = list(env.data_loader.indices)[:config.num_worlds]
        _files = [env.data_loader.dataset[i] for i in _idx]
        _ext, _exts = {}, {}
        for _w, _f in enumerate(_files):
            _b = os.path.basename(str(_f))
            if _b in _n2i:
                _j = _n2i[_b]
                _ext[_w] = _data['paths'][_j]
                _exts[_w] = _data['speeds'][_j]
        _K = install_external(builder, ego_indices, _ext, _exts,
                               origin_pos=_data.get('origin_pos'),
                               origin_heading=_data.get('origin_heading'),
                               last_valid_idx=(
                                   {_w: int(config.external_now_idx)
                                    for _w in _ext}
                                   if int(getattr(config, 'external_now_idx', 0))
                                   else None))
        logger.info(f'[EVAL] external candidates: {len(_ext)}/{config.num_worlds} '
                    f'worlds, K={_K}')
    agent = DiscreteIDCAgent(env, config, config.device, builder, ego_indices)

    all_files = []
    wm = WorldManager(env, builder, agent, all_files, config, logger, compute_density=False)
    wm.filter_initial(ego_indices)
    logger.info(f'[EVAL] {wm.good_count}/{config.num_worlds} worlds pass filter')

    logger.info(f'正在加载模型: {config.model_path}')
    agent.load(config.model_path)
    if getattr(config, 'check_tail', False):
        import numpy as _np
        import torch as _th
        from env.candidate_sampler import install_external
        _abs = env.sim.absolute_self_observation_tensor().to_torch().cpu().numpy()
        _maps = env.sim.map_observation_tensor().to_torch().cpu().numpy()
        _W = config.num_worlds

        def _mk(_turn):
            _P, _S = {}, {}
            for _w in range(_W):
                _px = float(_abs[_w, 0, 0])
                _py = float(_abs[_w, 0, 1])
                _ph = float(_abs[_w, 0, 7])
                _pts = []
                _th_ = _ph
                _x, _y = _px, _py
                for _i in range(91):
                    _pts.append([_x, _y])
                    if _i >= 40 and _turn:
                        _th_ += 0.35
                    _x += 1.0 * _np.cos(_th_)
                    _y += 1.0 * _np.sin(_th_)
                _P[_w] = _np.asarray(_pts, _np.float32)[None]
                _S[_w] = _np.full((1, 91), 10.0, _np.float32)
            return _P, _S

        def _states(_turn):
            _P, _S = _mk(_turn)
            install_external(builder, ego_indices, _P, _S, maps=_maps,
                             last_valid_idx={_w: 41 for _w in range(_W)})
            _st = builder.get_idc_observations_batch(
                ego_indices, path_indices=[0] * _W)
            return _np.stack([_st[_w] for _w in range(_W)])

        _sa = _states(False)
        _sb = _states(True)
        _ds = float(_np.max(_np.abs(_sa - _sb)))
        with _th.no_grad():
            _qa = agent._value(_th.from_numpy(_sa).float().to(config.device)).cpu().numpy()
            _qb = agent._value(_th.from_numpy(_sb).float().to(config.device)).cpu().numpy()
        _dq = float(_np.max(_np.abs(_qa - _qb)))
        _ok = (_ds < 1e-5 and _dq < 1e-4)
        logger.info(f'[0b] state max|diff|={_ds:.3e} critic max|diff|={_dq:.3e} '
                    f'-> {"PASS" if _ok else "FAIL"}')
        return
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

    _REPLAN = None
    if getattr(config, 'replan_candidates', ''):
        import numpy as _np
        from env.candidate_sampler import install_external
        _d = _np.load(config.replan_candidates, allow_pickle=True)
        _idx = list(env.data_loader.indices)[:config.num_worlds]
        _fs = [env.data_loader.dataset[i] for i in _idx]
        _n2i = {str(n): i for i, n in enumerate(_d['names'])}
        _op = _np.zeros((len(_fs), 2), _np.float32)
        _oh = _np.zeros(len(_fs), _np.float32)
        for _w, _f in enumerate(_fs):
            _k = _n2i.get(os.path.basename(str(_f)))
            if _k is not None:
                _op[_w] = _d['origin_pos'][_k]
                _oh[_w] = _d['origin_heading'][_k]
        _REPLAN = dict(frames=[int(x) for x in _d['frames']], data=_d,
                       files=_fs, n2i=_n2i, op=_op, oh=_oh)
        logger.info(f'[REPLAN] frames={_REPLAN["frames"]} '
                    f'matched={sum(1 for f in _fs if os.path.basename(str(f)) in _n2i)}'
                    f'/{len(_fs)}')

    def _dump_ref(tag):
        """打印 world0 候选的 pos/速度/朝向 + 派生特征，用于排查注入对齐。"""
        try:
            _w0, _a0 = 0, ego_indices[0]
            _cp = builder.candidate_paths[_w0][_a0]
            _p = _cp[0]
            logger.info(f'[DUMP-{tag}] 候选数={len(_cp)} '
                        f'keys={sorted(_p.keys())}')
            for _k in ('pos', 'speed', 'heading', 'curv_ahead', 'spd_ahead',
                       'road_ahead', 'road_dist'):
                if _k not in _p:
                    continue
                _v = np.asarray(_p[_k])
                if _v.ndim == 2:
                    logger.info(f'[DUMP-{tag}] {_k}: '
                                f'{np.round(_v[:3].reshape(-1), 2)} ... '
                                f'p20={np.round(_v[20].reshape(-1), 2)}')
                else:
                    logger.info(f'[DUMP-{tag}] {_k}: {np.round(_v[:4], 2)} ... '
                                f'v20={np.round(float(_v[20]), 2)}')
            _ep = builder.expert_pos[_w0][_a0].detach().cpu().numpy()
            logger.info(f'[DUMP-{tag}] expert p10={np.round(_ep[10, :2], 2)} '
                        f'p20={np.round(_ep[20, :2], 2)}')
            _abs = env.sim.absolute_self_observation_tensor().to_torch().cpu().numpy()
            logger.info(f'[DUMP-{tag}] ego_abs={np.round(_abs[0, 0, :2], 2)} '
                        f'yaw={float(_abs[0, 0, 7]):.3f}')
        except Exception as _e:
            logger.info(f'[DUMP-{tag}] 失败: {_e}')

    _RT = None
    if getattr(config, 'planner_runtime', ''):
        from env.planner_runtime import PlannerRuntime
        _RT = PlannerRuntime(
            config.planner_runtime, k=getattr(config, 'planner_k', 1),
            replan_every=getattr(config, 'planner_replan_every', 5),
            mode=getattr(config, 'planner_mode', 'planner'),
            device=getattr(config, 'device', 'cuda'),
            dump_path=getattr(config, 'planner_dump', ''))
        _RT.debug = True
        logger.info(f'[RT] 在线规划器: mode={config.planner_mode} '
                    f'every={config.planner_replan_every} k={config.planner_k}')

    def _rt_install(builder_, ego_idx, paths_, speeds_, tgt_):
        from env.candidate_sampler import install_external
        _maps = env.sim.map_observation_tensor().to_torch().cpu().numpy()
        install_external(builder_, ego_idx, paths_, speeds_,
                         target_pos=tgt_,
                         last_valid_idx={w: 41 for w in paths_},
                         blend_tail=True, tail_idx=41, maps=_maps)

    for epoch in range(config.epochs):
        obs = env.reset()
        for w in range(config.num_worlds):
            builder.reset_world_step(w, 0)
            agent.reset_world_state(w)
        builder.clear_cache()
        wm.begin_epoch()
        if _RT is not None:
            _RT.reset()
        if getattr(config, 'dump_ref', False):
            _dump_ref('internal')

        done_np = env.get_dones().cpu().numpy()
        for w in range(config.num_worlds):
            if w in wm.bad_worlds:
                continue
            a = ego_indices[w]
            if done_np[w, a] > 0.5:
                wm.reached_worlds.add(w)

        n_viz = max(int(getattr(config, 'traj_plot_max_worlds', 4)), 0)
        VIZ_WORLDS = list(wm.good_worlds)[:n_viz] if n_viz > 0 else []
        viz_list = [TrajectoryVisualizer(builder, w, ego_indices[w])
                    for w in VIZ_WORLDS]

        gif_worlds = _select_gif_worlds(config, VIZ_WORLDS)

        # 逐帧落盘，供离线画 storyboard（ego/伙伴位姿 + 事件）
        trace = {w: {'ego': [], 'partners': [], 'info': [],
                     'cands': [], 'sel': []} for w in (gif_worlds or [])}

        scorers = [PDMSScorer(config) for _ in range(config.num_worlds)]
        if getattr(config, 'dump_ref', False):
            try:
                _w0, _a0 = 0, ego_indices[0]
                _cp = builder.candidate_paths[_w0][_a0]
                logger.info(f'[DUMP] world0 候选数={len(_cp)} '
                            f'num_candidate_paths={builder.num_candidate_paths}')
                for _i, _p in enumerate(_cp[:2]):
                    _ps = _p['pos']
                    logger.info(f'[DUMP] cand{_i} keys={sorted(_p.keys())} '
                                f'lv={_p.get("last_valid_idx", "-")} '
                                f'p0={np.round(_ps[0],2)} p1={np.round(_ps[1],2)} '
                                f'p10={np.round(_ps[10],2)} p20={np.round(_ps[20],2)} '
                                f'sp0..3={np.round(_p["speed"][:4],2)} '
                                f'hd0..3={np.round(_p["heading"][:4],2)}')
                _ep = builder.expert_pos[_w0][_a0].detach().cpu().numpy()
                logger.info(f'[DUMP] expert p0={np.round(_ep[0,:2],2)} '
                            f'p10={np.round(_ep[10,:2],2)} '
                            f'p20={np.round(_ep[20,:2],2)}')
                _abs = env.sim.absolute_self_observation_tensor().to_torch().cpu().numpy()
                logger.info(f'[DUMP] ego_abs now={np.round(_abs[0,0,:2],2)} '
                            f'yaw={float(_abs[0,0,7]):.3f}')
            except Exception as _e:
                logger.info(f'[DUMP] 失败: {_e}')
        ref_start_agent = agent.DIM_EGO + agent.DIM_OTHERS + agent.DIM_VALIDITY
        temp_idx_pos = ref_start_agent + agent.DIM_REF_ERROR

        for step in range(max_step):
            logger.info(f'回合 {epoch+1}/{config.epochs}, 步数 {step+1}/{max_step}')

            if _REPLAN is not None and step in _REPLAN['frames']:
                _ri = _REPLAN['frames'].index(step)
                _ext, _exts = {}, {}
                for _w, _f in enumerate(_REPLAN['files']):
                    _k = _REPLAN['n2i'].get(os.path.basename(str(_f)))
                    if _k is not None:
                        _ext[_w] = _REPLAN['data']['paths'][_k][_ri]
                        _exts[_w] = _REPLAN['data']['speeds'][_k][_ri]
                _tp = None
                _maps = None
                try:
                    _abs = env.sim.absolute_self_observation_tensor().to_torch().cpu().numpy()
                    _nw = len(_REPLAN['files'])
                    _tp = ({_w: _abs[_w, 0, :2] for _w in range(_nw)}
                           if _abs.ndim == 3
                           else {_w: _abs[_w, :2] for _w in range(_nw)})
                    if step == _REPLAN['frames'][0]:
                        logger.info(f'[REPLAN] ego abs shape={_abs.shape} '
                                    f'w0={_tp.get(0)}')
                except Exception as _e:
                    logger.info(f'[REPLAN] ego pose unavailable: {_e}')
                try:
                    _maps = env.sim.map_observation_tensor().to_torch().cpu().numpy()
                except Exception as _e:
                    logger.info(f'[REPLAN] map unavailable: {_e}')
                install_external(builder, ego_indices, _ext, _exts,
                                 origin_pos=_REPLAN['op'],
                                 origin_heading=_REPLAN['oh'],
                                 target_pos=_tp,
                                 last_valid_idx=(
                                     {_w: int(config.external_tail_idx)
                                      for _w in _ext}
                                     if int(getattr(config, 'external_tail_idx', 0))
                                     else None),
                                 blend_tail=bool(getattr(config, 'external_blend_tail', False)),
                                 maps=_maps,
                                 tail_idx=int(getattr(config, 'external_tail_idx', 0)) or None)
                logger.info(f'[REPLAN] step {step} -> r{_ri} '
                            f'({len(_ext)} worlds)')
                if step == 0 and getattr(config, 'dump_ref', False):
                    _dump_ref('injected')

            if _RT is not None:
                _RT.observe(env, ego_indices)
                if step % max(_RT.replan_every, 1) == 0:
                    _RT.maybe_replan(step, env, builder, ego_indices, _rt_install)
                    if step == 0:
                        logger.info('[RT] 首次重规划完成')
                if step == 0 and getattr(config, 'dump_feat', ''):
                    try:
                        import numpy as _np2
                        _out = {}
                        _idx = list(env.data_loader.indices)[:config.num_worlds]
                        _files = [env.data_loader.dataset[i] for i in _idx]
                        for _w in range(config.num_worlds):
                            _a = int(ego_indices[_w])
                            _eg, (_x, _y, _h) = _RT._ego_feat(_w)
                            _ag, _am = _RT._agent_feat(env, _w, _a, _x, _y, _h)
                            _mp, _mm = _RT._map_feat(env, _w, _x, _y, _h)
                            _out[f'name{_w}'] = os.path.basename(str(_files[_w]))
                            _out[f'ego{_w}'] = _np2.asarray(_eg, _np2.float32)
                            _out[f'ag{_w}'] = _np2.asarray(_ag, _np2.float32)
                            _out[f'am{_w}'] = _np2.asarray(_am, _np2.float32)
                            _out[f'mp{_w}'] = _np2.asarray(_mp, _np2.float32)
                            _out[f'mm{_w}'] = _np2.asarray(_mm, _np2.float32)
                            _out[f'pose{_w}'] = _np2.asarray([_x, _y, _h], _np2.float32)
                        _np2.savez_compressed(config.dump_feat, **_out)
                        logger.info(f'[FEAT] dump {config.num_worlds} worlds -> {config.dump_feat}')
                    except Exception as _e:
                        logger.info(f'[FEAT] dump 失败: {_e}')

            _fi = int(getattr(config, 'force_path_index', -1))
            if _fi >= 0:
                path_indices = [_fi for _ in range(config.num_worlds)]
            else:
                path_indices = [0 for _ in range(config.num_worlds)]
            if _RT is not None and step % max(_RT.replan_every, 1) == 0:
                _RT.set_selected(path_indices[0])
            states = builder.get_idc_observations_batch(ego_indices, path_indices=path_indices)

            # Rollout PDMS — 仅在 step 0 对所有 good world 做一次前向推演
            if step == 0 and epoch == 0:
                rollout_scores = {}
                roll_scorer = RolloutPDMSScorer(agent, config, path_idx=0)
                for w in wm.good_worlds:
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

            # ---- storyboard trace：记录 ego 与伙伴的世界坐标位姿 ----
            for w in (gif_worlds or []):
                a = ego_indices[w]
                ex = float(abs_for_pdms[w, a, 0])
                ey = float(abs_for_pdms[w, a, 1])
                eh = float(abs_for_pdms[w, a, 7])
                trace[w]['ego'].append((ex, ey, eh))
                try:
                    _cc = [np.asarray((x['pos'].detach().cpu().numpy()
                                      if hasattr(x['pos'], 'detach') else x['pos']),
                                     np.float32)
                           for x in builder.candidate_paths[w][a]]
                except Exception:
                    _cc = []
                trace[w]['cands'].append(_cc)
                trace[w]['sel'].append(int(path_indices[w]) if 'path_indices' in dir() else 0)
                trace[w]['info'].append((float(info_np[w, a, 0]),
                                         float(info_np[w, a, 1]) + float(info_np[w, a, 2])))
                pn = partner_for_pdms[w, a]
                cos_h, sin_h = np.cos(eh), np.sin(eh)
                ps = []
                for i in range(pn.shape[0]):
                    sp, rx, ry, rh = (float(pn[i, 0]), float(pn[i, 1]),
                                      float(pn[i, 2]), float(pn[i, 3]))
                    if sp == 0.0 and rx == 0.0 and ry == 0.0:
                        continue
                    if not (np.isfinite(sp) and np.isfinite(rx)
                            and np.isfinite(ry) and np.isfinite(rh)):
                        continue
                    if abs(rx) > 1e3 or abs(ry) > 1e3 or abs(sp) > 1e2:
                        continue
                    ps.append((ex + rx * cos_h - ry * sin_h,
                               ey + rx * sin_h + ry * cos_h,
                               eh + rh, sp))
                trace[w]['partners'].append(ps)
            for w in wm.good_worlds:
                a = ego_indices[w]
                ego_x, ego_y = float(abs_for_pdms[w, a, 0]), float(abs_for_pdms[w, a, 1])
                ego_heading = float(abs_for_pdms[w, a, 7])
                ego_vel = float(rel_np[w, a, 0])
                partners = partner_for_pdms[w, a]
                off_road = float(info_np[w, a, 0]) > 0
                collision = (float(info_np[w, a, 1]) + float(info_np[w, a, 2])) > 0
                pid = 0
                # 用"投影进度索引"而不是步数时钟
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

            # 老的点线渲染只保留给 agent_pov；bird 视角改为 rollout 结束后
            # 用 storyboard 离线渲染（更快，也能看出车头朝向与尾迹）
            if (gif_enabled and config.gif_view_mode == 'agent_pov'
                    and step % recorder.fps == 0 and gif_worlds):
                _record_gif_frame(env, recorder, config, epoch, step,
                                  ego_indices, wm, gif_worlds)

        use_expert = getattr(config, 'use_expert_path', True)
        for viz in viz_list:
            if len(viz.actual_x) > 0:
                viz.save_plot(viz_dir, epoch + 1, use_expert_path=use_expert)
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
            plot_pdms_bar(bar_data,
                         os.path.join(pdms_plot_dir, f'pdms_bar_epoch{epoch+1}.png'),
                         title=f'Per-World Driving Score Epoch {epoch+1}')

    if trace:
        _dump_trace(trace, env, builder, config, ego_indices)
        if gif_enabled and config.gif_view_mode != 'agent_pov':
            _render_storyboards(trace, env, builder, config, ego_indices)
    if gif_enabled and config.gif_view_mode == 'agent_pov':
        recorder.save_all_gifs(custom_fps=config.gif_fps)
    logger.debug('评估完成')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='IDC-Bridge 评估脚本。所有默认值从 YAML 配置文件读取，CLI 参数可覆盖。')
    _default_cfg = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', 'configs', 'eval.yaml'))
    parser.add_argument('--config', type=str, default=_default_cfg,
                        help='YAML 配置文件路径')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Waymo tfrecord 数据目录')
    parser.add_argument('--model-path', type=str, required=True,
                        help='模型 checkpoint 路径')
    parser.add_argument('--check-tail', dest='check_tail', action='store_true', default=False)
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
    parser.add_argument('--cand-fixed', type=int, default=0,
                        help='>0 时安装固定候选集（3=±0.8/中心，5=±1.6/±0.8/中心）')
    parser.add_argument('--replan-candidates', type=str, default='',
                        help='滚动重规划 npz（含 frames 维）')
    parser.add_argument('--external-candidates', type=str, default='',
                        help='规划头 top-k 轨迹 npz（离线注入）')
    parser.add_argument('--external-now-idx', type=int, default=0,
                        help='外部参考里"当前时刻"所在的下标（>0 时通过 '
                             'last_valid_idx 传给 state builder；'
                             '前缀 11 点历史 + 规划 → 传 11）')
    parser.add_argument('--external-blend-tail', action='store_true',
                        help='规划窗之后接回原候选（沿路线），避免直线外推脱轨')
    parser.add_argument('--external-tail-idx', type=int, default=41,
                        help='从第几个点开始接回原候选（40 点规划 → 41）')
    parser.add_argument('--dump-ref', action='store_true',
                        help='打印 world0 的候选参考/专家/自车位姿，用于排查注入对齐')
    parser.add_argument('--planner-runtime', default='',
                        help='在线规划器 runtime：checkpoint 路径（按频率滚动重规划）')
    parser.add_argument('--planner-replan-every', type=int, default=5,
                        help='每多少步重规划一次（5 → 2 Hz）')
    parser.add_argument('--planner-mode', default='planner',
                        choices=['planner', 'route'],
                        help='route=把 env 内部参考原样重装（验收用）')
    parser.add_argument('--planner-k', type=int, default=1,
                        help='注入几条模态（IDC 未训过选择时用 1）')
    parser.add_argument('--planner-dump', default='',
                        help='把每次重规划的候选项/选中项/自车轨迹存成 npz（可视化用）')
    parser.add_argument('--dump-feat', default='',
                        help='在 step 0 把在线特征 dump 成 npz（P0 特征一致性用）')
    parser.add_argument('--force-path-index', type=int, default=-1,
                        help='>=0 时所有世界强制走第 i 条候选（A/B 对照用）')
    parser.add_argument('--gif-world-selection', type=str, default=None,
                        help='first / random')
    parser.add_argument('--gif-worlds', type=str, default='',
                        help='指定要录 gif 的 world 下标，逗号分隔，如 3,5')
    parser.add_argument('--traj-plot-max-worlds', type=int, default=4,
                        help='最多为多少个 world 录 gif（默认 4）')
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
        'check_tail': args.check_tail,
        'num_worlds': args.num_worlds,
        'epochs': args.epochs,
        'seed': args.seed,
        'device': args.device,
        'init_penalty': args.init_penalty,
        'max_penalty': args.max_penalty,
        'file_dir': args.file_dir,
        'gif_enabled': args.gif_enabled,
        'cand_fixed': args.cand_fixed,
        'force_path_index': args.force_path_index,
        'external_candidates': args.external_candidates,
        'external_now_idx': args.external_now_idx,
        'external_blend_tail': args.external_blend_tail,
        'external_tail_idx': args.external_tail_idx,
        'dump_ref': args.dump_ref,
        'planner_runtime': args.planner_runtime,
        'planner_replan_every': args.planner_replan_every,
        'planner_mode': args.planner_mode,
        'planner_k': args.planner_k,
        'planner_dump': args.planner_dump,
        'dump_feat': args.dump_feat,
        'replan_candidates': args.replan_candidates,
        'gif_view_mode': args.gif_view_mode,
        'gif_max_worlds': args.gif_max_worlds,
        'gif_zoom_radius': args.gif_zoom_radius,
        'gif_fps': args.gif_fps,
        'gif_record_interval': args.gif_record_interval,
        'gif_world_selection': args.gif_world_selection,
        'gif_worlds': args.gif_worlds,
        'traj_plot_max_worlds': args.traj_plot_max_worlds,
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









