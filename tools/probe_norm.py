"""用 RunningNorm 统计反推训练时的状态口径（世界坐标 vs 自车系）。"""
import sys

import numpy as np
import torch

sys.path.insert(0, '/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/tools')
import idc_policy as P

ck = torch.load(P.CKPT_DEFAULT, map_location='cpu', weights_only=False)
inn = ck['input_norm']
cnt = float(inn.get('count', 1.0)) or 1.0
mean = (inn['sum'] / cnt).numpy()
var = np.maximum(inn['sumsq'] / cnt - mean ** 2, 0)
std = np.sqrt(var + 1e-5)

names = (['ego_x', 'ego_y', 'ego_speed', 'ego_heading']
         + [f'o{k}_{n}' for k in range(8) for n in ('rx', 'ry', 'dphi', 'spd')]
         + [f'valid{k}' for k in range(8)]
         + ['dp', 'dphi', 'dv']
         + [f'la{k}_{n}' for k in range(3) for n in ('lat', 'dphi', 'road', 'spd')]
         + ['curv_ah', 'spd_ah', 'road_ah', 't', 'veh'])

print(f'{"idx":>4} {"name":>10} {"mean":>12} {"std":>12}')
for i in range(len(mean)):
    nm = names[i] if i < len(names) else f'?{i}'
    print(f'{i:>4} {nm:>10} {mean[i]:>12.3f} {std[i]:>12.3f}')

print()
print('关键判据:')
print(f'  ego_x/y 均值  = {mean[0]:.3f}, {mean[1]:.3f}   -> 0 表示训练已把自车放原点')
print(f'  others[0] rx/ry 均值 = {mean[4]:.3f}, {mean[5]:.3f}')
print(f'  others[0] rx/ry 标准差 = {std[4]:.3f}, {std[5]:.3f}')
print('  若 rx/ry 标准差很大（>30）说明训练用的是世界坐标；若 ~5-20 则是自车系相对量')
print(f'  dp 均值/标准差 = {mean[36]:.3f} / {std[36]:.3f}')
print(f'  t  均值/标准差 = {mean[63]:.3f} / {std[63]:.3f}')
