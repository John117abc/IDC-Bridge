#!/bin/bash
/mnt/2T_HDD/conda_envs/bench2drive_b2d/bin/python - <<'EOF'
import glob, os
from PIL import Image
fs = sorted(glob.glob('/mnt/2T_HDD/plant2_vector/video/frame_*.png'))
print('frames', len(fs))
if not fs:
    raise SystemExit
def make(sel, out, size=(640, 360), dur=90):
    imgs = [Image.open(f).convert('RGB').resize(size) for f in sel]
    if imgs:
        imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=dur, loop=0)
        print('saved', out, len(imgs))
# 全程（每 12 帧取 1）
make(fs[::12][:110], '/mnt/2T_HDD/plant2_vector/run_full.gif')
# 尾部 200 帧（卡死段，每秒 2 帧）
make(fs[-200::2], '/mnt/2T_HDD/plant2_vector/run_tail.gif', dur=140)
EOF
ls -la /mnt/2T_HDD/plant2_vector/*.gif 2>/dev/null
