"""核查 State v3 的 16 维新特征在真实数据里是否有有效信号。"""
import glob

import numpy as np

FS = sorted(glob.glob('/mnt/2T_HDD/plant2_vector/bc_night/*.npz'))
print('文件数', len(FS))
vru_cnt = vru_corr = light_any = cmd_left = remain_nz = 0
ego_dyn_nz = 0
n = 0
for f in FS[:40]:
    z = np.load(f)
    S = z['state']
    if S.shape[1] < 100:
        print('旧格式跳过', f)
        continue
    n += len(S)
    vru_cnt += int((S[:, 88] > 0).sum())          # 数量/4 > 0
    vru_corr += int((S[:, 87] > 0).sum())         # 是否在走廊
    light_any += int((S[:, 89:92].sum(1) > 0).sum())   # 红黄绿任一
    cmd_left += int((S[:, 96] > 0).sum())
    ego_dyn_nz += int((np.abs(S[:, 93:96]).sum(1) > 1e-6).sum())
    remain_nz += int((S[:, 99] > 0).sum())
print(f'样本 {n}')
print(f'  有行人计数   : {vru_cnt}  ({100*vru_cnt/max(n,1):.1f}%)')
print(f'  行人在走廊   : {vru_corr}  ({100*vru_corr/max(n,1):.1f}%)')
print(f'  有信号灯信息 : {light_any}  ({100*light_any/max(n,1):.1f}%)')
print(f'  左转指令     : {cmd_left}  ({100*cmd_left/max(n,1):.1f}%)')
print(f'  自车动态非零 : {ego_dyn_nz}  ({100*ego_dyn_nz/max(n,1):.1f}%)')
print(f'  剩余距离非零 : {remain_nz}  ({100*remain_nz/max(n,1):.1f}%)')
