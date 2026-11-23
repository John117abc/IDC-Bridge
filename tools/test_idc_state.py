"""自检：idc_state_carla 的维度与关键量是否合理。"""
import math
import sys

import numpy as np

sys.path.insert(0, '/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/tools')
import idc_state_carla as S

# 1) 一条简单直路（ego 在原点朝 +x，前方 40 m）
n = 41
pos = np.stack([np.linspace(0, 40, n), np.zeros(n)], 1).astype(np.float32)
path = dict(pos=pos,
            heading=np.zeros(n, np.float32),
            speed=np.full(n, 10.0, np.float32),
            road_dist=np.zeros(n, np.float32),
            curv_ahead=np.zeros(n, np.float32),
            spd_ahead=np.full(n, 10.0, np.float32),
            road_ahead=np.zeros(n, np.float32),
            last_valid_idx=n - 1)

# 2) 状态：自车在原点朝 +x、速度 8、前方 3 辆车
others = [[6.0, 0.0, 0.0, 8.0], [15.0, 1.5, 0.1, 9.0], [25.0, -1.0, 0.0, 7.0]]
st, t = S.build_state((0.0, 0.0, 0.0), 8.0, others, path, ref_idx=0)
print('dim            =', st.shape[0], '(期望 66)')
print('ref_idx        =', t, '(期望 0)')
print('ego[:6]        =', np.round(st[:6], 3))
print('others[0][:4]  =', np.round(st[6:10], 3))
print('validity       =', np.round(st[38:46], 1))
print('ref_err(3)     =', np.round(st[46:49], 3), ' <- [dp,dphi,dv]，dp 应≈0')
print('lookahead(12)  =', np.round(st[49:61], 3))
print('ahead(3)       =', np.round(st[61:64], 3))
print('temporal, veh  =', np.round(st[64:66], 3))

# 3) 横向偏移符号检查：把参考整体右移 2 m，dp 应为负（右偏）
path2 = dict(path)
path2['pos'] = (pos + np.array([0.0, 2.0], np.float32)).astype(np.float32)
st2, _ = S.build_state((0.0, 0.0, 0.0), 8.0, others, path2, ref_idx=0)
print('dp(参考右移2m) =', round(float(st2[46]), 3), ' <- 应为 -2 左右')

# 4) 索引投影：自车前进 10 m 后索引应推进到 ~10
st3, t3 = S.build_state((10.0, 0.0, 0.0), 8.0, others, path, ref_idx=0)
print('前进10m后 ref_idx =', t3, '(期望 ~10)')

# 5) 不倒退：prev_idx=20 时即使自车在起点也应保持 20
st4, t4 = S.build_state((0.0, 0.0, 0.0), 8.0, others, path, ref_idx=20)
print('不倒退检查 ref_idx =', t4, '(期望 20)')

# 6) build_path：8 点 → 41 点
plan = np.stack([np.arange(1, 9) * 5.0, np.zeros(8)], 1)
seg_a = np.stack([np.arange(0, 60, 1.0), np.zeros(60)], 1)
seg_b = seg_a + np.array([1.0, 0.0])
p = S.build_path(plan, (0.0, 0.0, 0.0), seg_a, seg_b)
print('build_path: pos', p['pos'].shape, 'speed[0:3]',
      np.round(p['speed'][:3], 2), 'road_dist[:3]', np.round(p['road_dist'][:3], 2))
st5, _ = S.build_state((0.0, 0.0, 0.0), 5.0, others, p, ref_idx=0)
print('用 build_path 组装 dim =', st5.shape[0])
