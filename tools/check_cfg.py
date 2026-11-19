import sys
sys.path.insert(0, '/gpudrive/baselines/IDC-Bridge/src')
from utils.config import build_config

KEYS = ('actor_type', 'bc_weight', 'curriculum_epochs', 'q_start_epoch', 'eps_anchor_epoch', 'path_select', 'run_tag', 'epochs', 'num_worlds', 'traj_plot_interval', 'actor_hidden', 'critic_hidden',
        'num_candidates', 'curriculum_epochs', 'q_start_epoch', 'path_select',
        'eps_start', 'eps_end', 'traj_plot_interval', 'run_tag', 'obs_norm',
        'epochs', 'num_worlds', 'window_size')
for cfg in ('pcont_tr', 'pcont_bc'):
    c = build_config(f'/gpudrive/baselines/IDC-Bridge/src/configs/{cfg}.yaml')
    print(cfg, {k: getattr(c, k, '<missing>') for k in KEYS})
