"""修正版：用正确 count 反推训练口径，并检查合成状态的归一化结果。"""
import sys

import numpy as np
import torch

sys.path.insert(0, '/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/tools')
import idc_policy as P
import idc_state_carla as S

ck = torch.load(P.CKPT_DEFAULT, map_location='cpu', weights_only=False)
inn = ck['input_norm']
print('input_norm keys:', list(inn.keys()) if hasattr(inn, 'keys') else type(inn))
cnt = float(inn.get('n', inn.get('count', 1.0)))
print('count =', cnt)
mean = (inn['sum'] / cnt).numpy()
var = np.maximum(inn['sumsq'] / cnt - mean ** 2, 0)
std = np.sqrt(var + 1e-5)

names = (['ego_x', 'ego_y', 'ego_speed', 'ego_heading']
         + [f'o{k}_{n}' for k in range(8) for n in ('rx', 'ry', 'dphi', 'spd')]
         + [f'valid{k}' for k in range(8)]
         + ['dp', 'dphi', 'dv']
         + [f'la{k}_{n}' for k in range(3) for n in ('lat', 'dphi', 'road', 'spd')]
         + ['curv_ah', 'spd_ah', 'road_ah', 't', 'veh'])

print(f'\n{"idx":>4} {"name":>10} {"mean":>10} {"std":>10}')
for i in list(range(0, 12)) + [36, 37, 44, 45, 46, 47, 48, 49, 59, 60, 61, 62, 63]:
    print(f'{i:>4} {names[i]:>10} {mean[i]:>10.3f} {std[i]:>10.3f}')

print('\n估算有效样本数（各通道 sum 与 mean 一致时）:', float(inn['sum'][2] / max(mean[2], 1e-6)))

# 合成状态归一化后的样子
n = 41
pos = np.stack([np.linspace(0, 40, n), np.zeros(n)], 1).astype(np.float32)
path = dict(pos=pos, heading=np.zeros(n, np.float32),
            speed=np.full(n, 10.0, np.float32),
            road_dist=np.zeros(n, np.float32), curv_ahead=np.zeros(n, np.float32),
            spd_ahead=np.full(n, 10.0, np.float32), road_ahead=np.zeros(n, np.float32),
            last_valid_idx=n - 1)
others = [[6.0, 0.0, 0.0, 8.0], [15.0, 1.5, 0.1, 9.0]]
st, _ = S.build_state((0.0, 0.0, 0.0), 8.0, others, path, ref_idx=0)
pol = P.IDCPolicy()
s_ego = P.to_ego_state(st, (0.0, 0.0, 0.0))
e = P.encode(s_ego).astype(np.float32)
z = (e - mean) / std
print('\n合成状态归一化后的 z 值（前 12 维）:')
for i in range(12):
    print(f'  {names[i]:>10} raw={e[i]:>8.2f}  z={z[i]:>10.2f}')
print('归一化后 |z| 最大值 = %.1f  （训练数据应在 ±5 内）' % np.abs(z).max())
print('超范围通道:', [names[i] for i in range(len(z)) if abs(z[i]) > 5][:8])
