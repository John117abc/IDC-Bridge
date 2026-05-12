from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.config import EnvConfig, RenderConfig

config = EnvConfig(
    ego_state=True,
    road_map_obs=True,
    partner_obs=True,
    lidar_obs=False,
    view_cone_obs=False,
    reward_type="distance_to_goal",
)

env = GPUDriveTorchEnv(config, data_dir="data/processed")