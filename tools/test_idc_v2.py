"""自检：idc_state_v2 的维度、边界与合理性。"""
import sys

import numpy as np

sys.path.insert(0, '/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/tools')
import idc_state_v2 as V

print('DIM_A =', V.DIM_A, '(期望 84)')
print('DIM_C =', V.DIM_C, '(期望 102)')

# 直路：前方 40 m
plan = np.stack([np.arange(1, 9) * 5.0, np.zeros(8)], 1)
path = V.path_from_plan(plan, (0.0, 0.0, 0.0))
print('path: n =', len(path['pos']), ' speed[:3] =', np.round(path['speed'][:3], 2),
      ' kappa max =', round(float(path['kappa'].max()), 4))

ego = dict(v=8.0, v_prev=7.8, omega=0.0, a_lat=0.0, steer_prev=0.0,
           acc_prev=0.5, pose=(0.0, 0.0, 0.0))
nbrs = [dict(x=6.0, y=0.0, h=0.0, v=8.0),
        dict(x=15.0, y=1.5, h=0.1, v=9.0),
        dict(x=25.0, y=-1.0, h=0.0, v=7.0)]
s = V.build_state_v2(ego, nbrs, path)
print('\n[A] 直路 dim =', s.shape[0])
print('  ego(8)      =', np.round(s[:8], 3))
print('  nbr0(6)     =', np.round(s[8:14], 3), ' <- rel_x>0, rel_y~0, in_corr=1')
print('  nbr1(6)     =', np.round(s[14:20], 3))
print('  validity    =', np.round(s[56:64], 2))
print('  ref_now(3)  =', np.round(s[64:67], 3), ' <- dp~0, dphi~0, dv~v-vref')
print('  lookahead   =', np.round(s[67:79], 3))
print('  ref_end(3)  =', np.round(s[79:82], 3))
print('  replan(2)   =', np.round(s[82:84], 3))
print('  |state| max =', round(float(np.abs(s).max()), 2), '(应 <=5)')

# 弯道：向右偏
plan_c = np.stack([np.arange(1, 9) * 5.0, np.cumsum(np.arange(1, 9) * 0.6)], 1)
path_c = V.path_from_plan(plan_c, (0.0, 0.0, 0.0))
sc = V.build_state_v2(ego, nbrs, path_c)
print('\n[B] 弯路 dim =', sc.shape[0])
print('  ref_end(3)  =', np.round(sc[79:82], 3), ' <- lat 应为正（向右）')
print('  kappa 最大  =', round(float(path_c['kappa'].max()), 3),
      '-> lookahead κ 项 =', np.round(sc[67:79][3::4], 2))

# 阶段 C
modes = [path, path_c, path, path_c, path, path_c]
s2 = V.build_state_v2(ego, nbrs, path, modes=modes)
print('\n[C] 带模态 dim =', s2.shape[0])
print('  mode 块 =', np.round(s2[84:102], 3))

# 周车全部在后方 → validity 应为 0
nbrs_back = [dict(x=-5.0 - i, y=0.0, h=0.0, v=5.0) for i in range(3)]
sb = V.build_state_v2(ego, nbrs_back, path)
print('\n[D] 后方周车 validity =', np.round(sb[56:64], 2),
      ' ttc 项 =', np.round(sb[8:14][4], 2))

# 空周车
s0 = V.build_state_v2(ego, [], path)
print('[E] 无周车 validity 和 =', float(s0[56:64].sum()))
