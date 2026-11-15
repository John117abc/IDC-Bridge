"""探查 map_observation_tensor 的列含义（MapObservation 是线段不是点）。"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import numpy as np  # noqa: E402


def main():
    from gpudrive.env.dataset import SceneDataLoader
    from gpudrive.env.env_torch import GPUDriveTorchEnv

    from env.env_utils import get_env_config

    loader = SceneDataLoader(root="/data/data_json/training", batch_size=4,
                             dataset_size=4, sample_with_replacement=False,
                             shuffle=True, seed=0)
    env = GPUDriveTorchEnv(config=get_env_config(), data_loader=loader,
                           max_cont_agents=1, device="cuda",
                           action_type="continuous")
    env.reset()
    m = env.sim.map_observation_tensor().to_torch().cpu().numpy()
    print("shape:", m.shape)
    rows = m[0]
    nz = rows[np.abs(rows).sum(1) > 1e-6]
    print(f"world0 非零行 {len(nz)}/{len(rows)}")
    print("列统计（world0 非零行）:")
    names = ["x", "y", "c2", "c3", "c4", "c5", "c6", "c7", "c8"]
    for i in range(rows.shape[1]):
        c = nz[:, i]
        uniq = len(np.unique(np.round(c, 3)))
        print(f"  [{i}] {names[i]:3s} min={c.min():9.3f} max={c.max():9.3f} "
              f"mean={c.mean():9.3f}  uniq={uniq}")
    print()
    print("前 8 行原始值:")
    for r in nz[:8]:
        print("  " + " ".join(f"{v:8.3f}" for v in r))
    print()
    # 按 type 分组计数（假设 type 在 6）
    for ti in (5, 6, 7):
        vals, cnt = np.unique(nz[:, ti], return_counts=True)
        if len(vals) <= 12:
            print(f"列[{ti}] 取值: {dict(zip(np.round(vals,2).tolist(), cnt.tolist()))}")


if __name__ == "__main__":
    main()
