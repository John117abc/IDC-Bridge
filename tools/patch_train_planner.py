"""给 idc-train.py 接上在线规划器（每 N 步注入参考），用于重训 IDC。"""
import ast

f = '/gpudrive/baselines/IDC-Bridge/src/scripts/train/idc-train.py'
src = open(f, encoding='utf-8-sig').read()
n = 0

# ---------- 1) CLI 参数 ----------
anchor_cli = "    parser.add_argument('--state-mode', type=str, default=None,"
add_cli = """    parser.add_argument('--planner-runtime', default='',
                        help='在线规划器 checkpoint（每 N 步注入参考）')
    parser.add_argument('--planner-replan-every', type=int, default=5)
    parser.add_argument('--planner-k', type=int, default=1)
""" + anchor_cli
if '--planner-runtime' not in src:
    assert anchor_cli in src, '找不到 --state-mode 锚点'
    src = src.replace(anchor_cli, add_cli, 1)
    n += 1

# ---------- 2) 模块级持有（在 logger 定义之后）----------
anchor_log = "logger = get_logger('idc-train')"
add_log = anchor_log + """
_PLANNER_CACHE = {'rt': None, 'install': None}
"""
if '_PLANNER_CACHE' not in src:
    assert anchor_log in src, '找不到 logger 锚点'
    src = src.replace(anchor_log, add_log, 1)
    n += 1

# ---------- 3) 步循环里注入重规划 ----------
anchor_step = "            partner_np_step = env.sim.partner_observations_tensor().to_torch().cpu().numpy()"
add_step = anchor_step + """

            # ---- 在线规划器：每 N 步把规划轨迹注入为候选参考（重训 IDC 用）----
            if getattr(config, 'planner_runtime', ''):
                if step == 0 or _PLANNER_CACHE['rt'] is None:
                    from env.planner_runtime import PlannerRuntime
                    from env.candidate_sampler import install_external
                    _rt = PlannerRuntime(
                        config.planner_runtime,
                        k=int(getattr(config, 'planner_k', 1)),
                        replan_every=int(getattr(config, 'planner_replan_every', 5)),
                        device=getattr(config, 'device', 'cuda'),
                        mode='planner')
                    _rt.debug = True
                    _rt.reset()

                    def _planner_install(b, ego_idx, paths, speeds, tgt,
                                         _env=env):
                        _maps = _env.sim.map_observation_tensor().to_torch().cpu().numpy()
                        install_external(b, ego_idx, paths, speeds,
                                         target_pos=tgt,
                                         last_valid_idx={w: 41 for w in paths},
                                         blend_tail=True, tail_idx=41, maps=_maps)
                    _PLANNER_CACHE['rt'] = _rt
                    _PLANNER_CACHE['install'] = _planner_install
                try:
                    _rt = _PLANNER_CACHE['rt']
                    _rt.observe(env, ego_indices)
                    if step % max(int(getattr(config, 'planner_replan_every', 5)), 1) == 0:
                        _rt.maybe_replan(step, env, builder, ego_indices,
                                         _PLANNER_CACHE['install'])
                except Exception as _re:
                    logger.warning(f'[RT] 重规划失败: {_re}')"""
if '在线规划器：每 N 步' not in src:
    assert anchor_step in src, '找不到 partner_np_step 锚点'
    src = src.replace(anchor_step, add_step, 1)
    n += 1

open(f, 'w', encoding='utf-8').write(src.lstrip('\ufeff'))
ast.parse(open(f, encoding='utf-8').read())
s = open(f, encoding='utf-8').read()
print('PATCH OK 处数 =', n,
      '| cli =', '--planner-runtime' in s,
      '| cache =', '_PLANNER_CACHE' in s,
      '| loop =', '在线规划器：每 N 步' in s)
