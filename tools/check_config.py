"""校验 train.yaml(=_base base.yaml) 经 build_config 后能暴露 idc_agent 需要的键。"""
import os
import sys
from types import SimpleNamespace

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'src', 'utils'))

# 直接执行 config.py，绕开 utils/__init__ 里对 gpudrive 的间接依赖
import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location(
    'idc_config', os.path.join(ROOT, 'src/utils/config.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

KEYS = ['dt', 'horizon', 'batch_size', 'pim_interval', 'gamma', 'window_size',
        'epochs', 'num_worlds', 'num_worlds', 'hidden_dim',
        'pos_err_weight', 'speed_err_weight', 'heading_err_weight',
        'steer_cost_weight', 'acc_cost_weight',
        'lookahead_pos_weight', 'lookahead_heading_weight',
        'progress_weight', 'ttc_weight', 'jerk_weight', 'jerk_max',
        'road_weight', 'init_penalty', 'max_penalty', 'amplifier_c',
        'utility_clamp', 'penalty_clamp', 'use_is_weights',
        'road_in_penalty', 'penalty_schedule', 'gep_amplifier',
        'veh_safe_ellipse', 'veh_safe_long', 'veh_safe_lat',
        'D_veh_safe', 'D_road_safe', 'half_length', 'half_width',
        'noise_std', 'noise_decay_rate', 'noise_std_min', 'bc_weight',
        'wheelbase', 'lr_ratio', 'v_max', 'fix_speed', 'fix_heading',
        'no_sign', 'no_road_penalty', 'no_veh_penalty',
        'transformer_d_model', 'transformer_nhead', 'transformer_num_layers',
        'transformer_dropout', 'lr_actor_max', 'lr_actor_min',
        'lr_critic_max', 'lr_critic_min', 'buffer_capacity',
        'state_mode', 'ref_mode', 'ref_window', 'obs_norm',
        'pos_err_clamp', 'lat_err_clamp', 'pev_interval']

cfg = mod.build_config(os.path.join(ROOT, 'src/configs/train.yaml'))
missing = [k for k in KEYS if not hasattr(cfg, k)]
print('MISSING:', missing if missing else 'none')

show = ['road_weight', 'utility_clamp', 'penalty_clamp', 'penalty_schedule',
        'gep_amplifier', 'use_is_weights', 'road_in_penalty', 'veh_safe_ellipse',
        'veh_safe_long', 'veh_safe_lat', 'steer_cost_weight', 'acc_cost_weight',
        'noise_std', 'horizon', 'dt', 'gamma', 'wheelbase', 'epochs', 'num_worlds',
        'state_mode', 'ref_mode', 'ref_window', 'obs_norm',
        'pos_err_clamp', 'lat_err_clamp', 'pev_interval', 'pim_interval']
for k in show:
    print(f'  {k:20s} = {getattr(cfg, k, "<MISSING>")}')

assert not missing, f'配置缺键: {missing}'
print('config OK')
