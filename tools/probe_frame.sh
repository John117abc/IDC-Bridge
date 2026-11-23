#!/bin/bash
set -u
C=idc_hist
echo "=== GPUDrive 里的坐标变换字段 ==="
docker exec $C bash -lc "grep -rnE 'global_center|center_x|sdc_angle|global_angle|normalize|mean_x|offset' /gpudrive/gpudrive/env/env_torch.py /gpudrive/gpudrive/env/dataset.py 2>/dev/null | head -12"
echo
echo "=== simulator 里的字段 ==="
docker exec $C bash -lc "grep -rnE 'global_center|center|angle' /gpudrive/gpudrive/simulator.py 2>/dev/null | head -12"
echo
echo "=== 运行时对象属性（模拟器）==="
docker exec $C bash -lc 'cd /gpudrive/baselines/IDC-Bridge && /gpudrive/.venv/bin/python - <<PY
import sys, os
sys.path.insert(0, "src")
from env.env_utils import get_env_config
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.dataset import SceneDataLoader
loader = SceneDataLoader(root="/data/data_json/training", batch_size=1, dataset_size=1,
                         sample_with_replacement=False, shuffle=False, seed=0)
env = GPUDriveTorchEnv(config=get_env_config(), data_loader=loader, max_cont_agents=8,
                       device="cuda", action_type="continuous")
sim = env.sim
for name in dir(sim):
    if any(k in name.lower() for k in ("center", "angle", "origin", "global", "offset")):
        try:
            v = getattr(sim, name)
            print(name, "=", (v if not hasattr(v, "shape") else tuple(v.shape)))
        except Exception:
            pass
PY' 2>&1 | tail -12
