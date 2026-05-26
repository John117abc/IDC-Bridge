import json

import torch
from gpudrive.env.config import EnvConfig


def extend_action_to_3d(actions_2d):
    batch, agents, _ = actions_2d.shape
    zeros = torch.zeros((batch, agents, 1), dtype=actions_2d.dtype, device=actions_2d.device)
    return torch.cat([actions_2d, zeros], dim=-1)


def get_env_config(**overrides):
    config = dict(
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
    )
    config.update(overrides)
    return EnvConfig(**config)


def load_scenes(env):
    scenes = []
    for fpath in env.data_batch:
        with open(fpath, 'r') as f:
            scenes.append(json.load(f))
    return scenes


def get_ego_indices(env, num_worlds):
    ego_indices = []
    for w in range(num_worlds):
        controllable = env.cont_agent_mask[w].nonzero(as_tuple=True)[0]
        ego_indices.append(controllable[0].item() if len(controllable) > 0 else 0)
    return ego_indices
