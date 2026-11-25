"""生成 gif：保持原始分辨率（或放大），避免低分辨率糊掉"""
import glob
import os
import sys

from PIL import Image

src, out = sys.argv[1], sys.argv[2]
scale = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
step = int(sys.argv[4]) if len(sys.argv) > 4 else 2
maxn = int(sys.argv[5]) if len(sys.argv) > 5 else 200

fs = sorted(glob.glob(os.path.join(src, 'frame_*.png')))
if not fs:
    print('没有帧', src)
    sys.exit(1)

sel = fs[::step][:maxn]
imgs = []
for f in sel:
    im = Image.open(f).convert('RGB')
    if scale != 1.0:
        im = im.resize((int(im.width * scale), int(im.height * scale)),
                       Image.LANCZOS)
    imgs.append(im)
imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=110, loop=0,
             optimize=False)
print(f'gif {os.path.basename(out)} 帧数 {len(imgs)} 尺寸 {imgs[0].size}')
