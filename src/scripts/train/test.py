import os
from pathlib import Path
import torch
import mediapy
import imageio
import json
import sys
import numpy as np

from gpudrive.env.config import EnvConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.visualize.utils import img_from_fig
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.datatypes.observation import (
    LocalEgoState,
    GlobalEgoState,
    PartnerObs,
)

sys.path.insert(0, '/workspace/idc/src')
from env.idc_state_builder_bak import GPUDriveObservationBuilder
MAX_NUM_OBJECTS = 1  # Maximum number of objects in the scene we control
NUM_WORLDS = 2  # Number of parallel environments
UNIQUE_SCENES = 2 # Number of unique scenes

device = 'cuda' # for simplicity purposes in notebook we use cpu, note that the simulator is optimized for GPU so use cuda if possible


# 方式2：多维动作 (num_worlds, max_agents, action_dim)
def create_forward_actions_multidim(num_worlds, max_agents, action_dim=3):
    """
    创建三维动作张量
    形状: (num_worlds, max_agents, action_dim)
    """
    # 向前走的动作：[转向=0, 加速度=0.5, 头部倾斜=0]
    forward_action = torch.tensor([0.5, 0.0, 0.0], dtype=torch.float32)
    
    # 扩展到所有智能体
    actions = forward_action.reshape(1, 1, -1).expand(num_worlds, max_agents, -1).clone()
    return actions


env_config = EnvConfig(
        ego_state=True,
        road_map_obs=True,
        partner_obs=True,
        norm_obs=True,
        reward_type="weighted_combination",
        dynamics_model="classic",
        max_controlled_agents=1,
        collision_behavior="ignore",
        dist_to_goal_threshold=2.0,
        polyline_reduction_threshold=0.1,
        obs_radius=50.0,
        episode_len=100,
        max_num_agents_in_scene=8,
        roadgraph_top_k = 100,
)


# Make dataloader
data_loader = SceneDataLoader(
    root="data/processed/examples", # Path to the dataset
    batch_size=NUM_WORLDS, # Batch size, you want this to be equal to the number of worlds (envs) so that every world receives a different scene
    dataset_size=UNIQUE_SCENES, # Total number of different scenes we want to use
    sample_with_replacement=False,
    seed=42,
    shuffle=True,
)

# 加载环境
env = GPUDriveTorchEnv(
    config=env_config,
    data_loader=data_loader,
    max_cont_agents=MAX_NUM_OBJECTS, # Maximum number of agents to control per scenario
    device=device,
    action_type="continuous",
)

scene_files = env.data_batch  # 根据实际获取
scenes = []
for f in scene_files:
    with open(f) as fp:
        scenes.append(json.load(fp))

builder = GPUDriveObservationBuilder(env, scenes)
for world in range(NUM_WORLDS):
    builder.reset_world_step(world, 0)

obs = env.reset()


frames = {f"env_{i}": [] for i in range(NUM_WORLDS)}


# 获取原始 tensor（具体方法名可能略有不同，请根据实际环境调整）
# 常见的方法名：
self_obs_tensor = env.sim.absolute_self_observation_tensor().to_torch()        # (num_worlds, max_agents, 8)
partner_obs_tensor = env.sim.partner_observations_tensor().to_torch()  # (num_worlds, max_agents, max_agents-1, 9)
map_obs_tensor = env.sim.map_observation_tensor().to_torch()          # (num_worlds, max_agents, top_k, 13)

print(f"self_obs_tensor shape: {self_obs_tensor.shape}")

# 缓慢向前
action_slow_forward = (
    np.array([0.0], dtype=np.float32),   # 转向角 = 0（直行）
    np.array([0.5], dtype=np.float32),   # 加速度 = 0.5（缓慢向前）
    np.array([0.0], dtype=np.float32)    # 头部倾斜 = 0
)

for t in range(env_config.episode_len):

    # 创建批量动作
    actions = create_forward_actions_multidim(NUM_WORLDS, MAX_NUM_OBJECTS, action_dim=3)
    
    # Step the environment
    env.step_dynamics(actions)
    obs = env.get_obs()
    reward = env.get_rewards()
    done = env.get_dones()
    world_idx = 0
    agent_idx = 7
    # Render the environment
    if t % 5 == 0:
        imgs = env.vis.plot_simulator_state(
            env_indices=list(range(NUM_WORLDS)),
            time_steps=[t] * NUM_WORLDS,
            zoom_radius=70,
        )
        for i in range(NUM_WORLDS):
            frames[f"env_{i}"].append(img_from_fig(imgs[i]))

    if done.all():
        break

# 每个环境的帧保存为单独的 GIF
print("开始保存")
save_dir = "/workspace/idc/gifs"
os.makedirs(save_dir, exist_ok=True)

for env_name, frame_list in frames.items():
    path = os.path.join(save_dir, f"rollout_{env_name}.gif")
    imageio.mimsave(path, frame_list, fps=5)
