"""自检：CARLA 侧 IDC 能否加载权重、构造 84 维状态、输出合理动作。"""
import sys

import numpy as np

sys.path.insert(0, '/gpudrive/baselines/IDC-Bridge/tools')
import frames as F  # noqa: E402
import idc_carla as IC  # noqa: E402

CK = sys.argv[1] if len(sys.argv) > 1 else \
    '/workspace/data/checkpoints/20261119/idc-waymo-v1.0_examples_041839_episode=475.pth'

# 一条直路参考（canonical 世界坐标）
n = 41
pos = np.stack([np.linspace(0, 40, n), np.zeros(n)], 1).astype(np.float32)
path = dict(pos=pos,
            heading=np.zeros(n, np.float32),
            speed=np.full(n, 8.0, np.float32),
            road_dist=np.zeros(n, np.float32),
            curv_ahead=np.zeros(n, np.float32),
            spd_ahead=np.full(n, 8.0, np.float32),
            road_ahead=np.zeros(n, np.float32),
            last_valid_idx=n - 1)

ego = (0.0, 0.0, 0.0)
nbrs = [dict(x=10.0, y=0.5, h=0.0, v=6.0), dict(x=20.0, y=-1.5, h=0.0, v=7.0)]

st, t = IC.make_state84(ego, 8.0, nbrs, path, ref_idx=0)
print('state84 =', st.shape, ' ref_idx =', t, ' |x|max = %.2f' % float(np.abs(st).max()))

pol = IC.IDCCarlaPolicy(CK)
for k in range(3):
    d, a = pol.act(st)
    steer, thr, brk = IC.to_carla_control(d, a)
    print(f'step{k}: delta={d:+.3f} rad  acc={a:+.2f}  → steer={steer:+.3f} '
          f'thr={thr:.2f} brk={brk:.2f}')
