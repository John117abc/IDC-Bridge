import os
from pathlib import Path
import torch
import mediapy
import imageio
import json
import sys

from gpudrive.env.config import EnvConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.visualize.utils import img_from_fig
from gpudrive.env.dataset import SceneDataLoader

sys.path.insert(0, '/workspace/idc/src')
from env.idc_state_builder import GPUDriveObservationBuilder
MAX_NUM_OBJECTS = 64  # Maximum number of objects in the scene we control
NUM_WORLDS = 2  # Number of parallel environments
UNIQUE_SCENES = 2 # Number of unique scenes

device = 'cuda' # for simplicity purposes in notebook we use cpu, note that the simulator is optimized for GPU so use cuda if possible

env_config = EnvConfig(
    steer_actions = torch.round(
        torch.linspace(-1.0, 1.0, 3), decimals=3),
    accel_actions = torch.round(
        torch.linspace(-3, 3, 3), decimals=3
    )
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

for t in range(env_config.episode_len):

    # Sample random actions
    rand_action = torch.Tensor(
        [[env.action_space.sample() for _ in range(MAX_NUM_OBJECTS * NUM_WORLDS)]]
    ).reshape(NUM_WORLDS, MAX_NUM_OBJECTS)

    # Step the environment
    print(f'动作形状{rand_action.shape}')
    env.step_dynamics(rand_action)

    obs = env.get_obs()
    reward = env.get_rewards()
    done = env.get_dones()
    world_idx = 0
    agent_idx = 7
    net_state, s_road, s_ref_raw, s_ref_error, s_other = builder.get_idc_observation(
        world_idx, agent_idx, perceived_distance=30.0
    )
    print(f'net_state:{net_state}')
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
