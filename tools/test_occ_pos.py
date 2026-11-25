"""验证位置编码：同一组块交换应产生不同输出；并检查 patch 坐标是否合理"""
import sys

import torch

R = '/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge'
sys.path.insert(0, R + '/src')
sys.path.insert(0, R + '/tools')
import importlib                                       # noqa: E402

import train_planner                                   # noqa: E402
importlib.reload(train_planner)
from train_planner import Planner                      # noqa: E402

m = Planner(K=6, use_occ=True, occ_pos=True)
print('occ_xy 前 3 个 patch 中心（米）:')
xy = m.occ_xy
for k in (0, 1, 8, 9, 35, 63):
    print(f'  patch {k:2d}  x={xy[k,0]:+6.2f}  y={xy[k,1]:+6.2f}')
print('occ_pos_mlp 参数量:', sum(p.numel() for p in m.occ_pos_mlp.parameters()))

# 交换两个块，输出应不同
m.emit_logits = True
B = 1
ego = torch.randn(B, 23); ag = torch.randn(B, 32, 28); am = torch.ones(B, 32)
mp = torch.randn(B, 152, 7); mm = torch.ones(B, 152)
occ = torch.zeros(B, 3, 64, 64)
occ[0, 2, 40:48, 24:32] = 1.0          # 走廊块 (br=5, bc=3)
occ2 = occ.clone()
occ2[0, 2, 40:48, 24:32] = 0.0
occ2[0, 2, 40:48, 0:8] = 1.0           # 移到路边块 (br=5, bc=0)
with torch.no_grad():
    p1, l1 = m(ego, ag, am, mp, mm, occ=occ)
    p2, l2 = m(ego, ag, am, mp, mm, occ=occ2)
print(f'\n随机权重下：max|Δpred|={float((p1-p2).abs().max()):.6f}  '
      f'max|Δlogit|={float((l1-l2).abs().max()):.6f}')
print('（只要 >0 即说明结构上不再是置换不变）')
