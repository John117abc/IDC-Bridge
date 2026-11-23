#!/bin/bash
set -u
PY=/mnt/2T_HDD/conda_envs/bench2drive_b2d/bin/python
D=/mnt/2T_HDD/plant2_vector
$PY - <<'EOF'
import glob, os
from PIL import Image

D = '/mnt/2T_HDD/plant2_vector'
fs = sorted(glob.glob(os.path.join(D, 'video', 'frame_*.png')))
print('总帧数', len(fs))
if not fs:
    raise SystemExit('没有录像帧')

# 每 4 帧取一帧，最多 120 帧
sel = fs[::4][:120]
imgs = []
for f in sel:
    im = Image.open(f).convert('RGB')
    im = im.resize((im.width // 2, im.height // 2))
    imgs.append(im)
imgs[0].save(os.path.join(D, 'carla_camera.gif'), save_all=True,
             append_images=imgs[1:], duration=120, loop=0)
print('gif 保存', os.path.join(D, 'carla_camera.gif'), '帧数', len(imgs))

# 关键帧：开头 / 中段 / 末尾
picks = [0, len(fs) // 2, len(fs) - 1]
for k, i in enumerate(picks):
    dst = os.path.join(D, f'carla_key_{k}.png')
    Image.open(fs[i]).convert('RGB').save(dst)
    print('关键帧', i, '->', dst)
EOF
