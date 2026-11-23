"""自检：IDC 策略能否载入 checkpoint 并前向出动作。"""
import sys

import numpy as np

sys.path.insert(0, '/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/tools')
import idc_state_carla as S
import idc_policy as P

n = 41
pos = np.stack([np.linspace(0, 40, n), np.zeros(n)], 1).astype(np.float32)
path = dict(pos=pos, heading=np.zeros(n, np.float32),
            speed=np.full(n, 10.0, np.float32),
            road_dist=np.zeros(n, np.float32),
            curv_ahead=np.zeros(n, np.float32),
            spd_ahead=np.full(n, 10.0, np.float32),
            road_ahead=np.zeros(n, np.float32),
            last_valid_idx=n - 1)

pol = P.IDCPolicy()
print('state_dim/state_mode/ref_mode =', pol.state_dim, pol.state_mode, pol.ref_mode)
print('RunningNorm count =', float(pol.norm.count), ' mean[:3] =',
      np.round(pol.norm.mean[:3].cpu().numpy(), 3),
      ' std[:3] =', np.round(pol.norm.std[:3].cpu().numpy(), 3))

others = [[6.0, 0.0, 0.0, 8.0], [15.0, 1.5, 0.1, 9.0]]
st, t = S.build_state((0.0, 0.0, 0.0), 8.0, others, path, ref_idx=0)
print('state dim =', st.shape)

for step in range(3):
    steer, acc, delta = pol.act(st, (0.0, 0.0, 0.0))
    print(f'step {step}: steer_norm={steer:+.3f}  delta={delta:+.3f} rad  acc={acc:+.2f} m/s²')
