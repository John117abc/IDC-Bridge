"""占据栅格示意图 v3：前方朝上、左侧为 -y，带米标尺"""
import glob
import os

import numpy as np
from PIL import Image, ImageDraw

D = '/mnt/2T_HDD/plant2_vector/occupancy'
fs = sorted(glob.glob(os.path.join(D, 'ConstructionObstacle__*.npz')))

best = None
for f in fs[:40]:
    z = np.load(f)
    occ = z['occ']
    s = occ[:, 2].sum(axis=(1, 2))
    i = int(np.argmax(s))
    if best is None or s[i] > best[0]:
        best = (int(s[i]), f, i, occ)

s, f, i, occ = best
fr = occ[i]
print(f'样本 {os.path.basename(f)} 帧 {i} 静态格 {s}')

SCALE = 6
col = np.zeros((64, 64, 3), np.uint8)
col[fr[0] > 0] = [235, 70, 70]
col[fr[1] > 0] = [70, 220, 90]
col[fr[2] > 0] = [80, 140, 255]

# 转成显示坐标：横轴 = y（左负右正），纵轴 = x（上为前）
disp = np.transpose(col, (1, 0, 2))       # (y, x)
disp = np.flipud(disp)                    # x 增大向上
big = np.repeat(np.repeat(disp, SCALE, axis=0), SCALE, axis=1)
H, W = big.shape[:2]
img = Image.fromarray(big)
d = ImageDraw.Draw(img)

# 网格：每 2 m
for k in range(0, 65, 4):
    d.line([(k * SCALE, 0), (k * SCALE, H)], fill=(38, 38, 44))
    d.line([(0, k * SCALE), (W, k * SCALE)], fill=(38, 38, 44))

# 自车：y=0 → 列 32；x=0 → 原始行 16 → 翻转后行 63-16=47
cx = 32 * SCALE
cy = (63 - 16) * SCALE
d.polygon([(cx, cy - 16), (cx - 10, cy + 8), (cx + 10, cy + 8)], fill=(250, 250, 250))
d.rectangle([cx - 5, cy + 6, cx + 5, cy + 15], outline=(250, 250, 250), width=2)

d.text((6, 4), 'red=vehicles  green=VRU  blue=static props', fill=(210, 210, 210))
d.text((6, 17), 'forward = up   |   0.5 m/cell   |   +-16 m lateral',
       fill=(150, 150, 150))
for m in (0, 8, 16, 24):
    yy = (63 - int((m + 8) / 0.5)) * SCALE
    d.text((2, min(max(yy, 2), H - 10)), f'{m}m', fill=(120, 120, 120))

out = '/mnt/2T_HDD/plant2_vector/occ_sample_v3.png'
img.save(out)
print('saved', out, img.size)
