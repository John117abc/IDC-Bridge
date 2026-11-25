"""合成 gif：左=车头相机，右=自车系规划面板（栅格/周车/route/6 模态/选中）

兼容两种 dbg：
  - ctrl (N,3) = steer/thr/brk
  - route (N,24,4) 或 (N,24,2)
"""
import argparse
import glob
import os
import re

import numpy as np
from PIL import Image, ImageDraw

PY_M, PX_M = 13.0, 7.0
XMIN, XMAX = -5.0, 45.0
YMIN, YMAX = -12.0, 12.0
PANEL_W, PANEL_H = 640, 360
CX, CY = 170.0, 340.0


def to_px(x, y):
    return (CX + y * PY_M, CY - (x - XMIN) * PX_M)


def draw_panel(d, e):
    d.rectangle([0, 0, PANEL_W, PANEL_H], fill=(16, 16, 20))
    for m in range(int(XMIN), int(XMAX) + 1, 5):
        p0, p1 = to_px(m, YMIN), to_px(m, YMAX)
        d.line([p0, p1], fill=(44, 44, 50))
        d.text((4, min(max(p0[1] - 6, 2), PANEL_H - 12)), f'{m}m', fill=(110, 110, 110))
    for n in range(int(YMIN), int(YMAX) + 1, 5):
        d.line([to_px(XMIN, n), to_px(XMAX, n)], fill=(44, 44, 50))

    occ = e.get('occ')
    if occ is not None and np.asarray(occ).shape == (3, 64, 64):
        o = np.asarray(occ)
        for ch, col in ((0, (65, 65, 85)), (1, (60, 170, 80)), (2, (210, 55, 55))):
            rr, cc = np.nonzero(o[ch])
            for r, c in zip(rr, cc):
                x = -8 + (r + 0.5) * 0.5
                y = -16 + (c + 0.5) * 0.5
                px, py = to_px(x, y)
                d.rectangle([px - PY_M * 0.24, py - PX_M * 0.24,
                             px + PY_M * 0.24, py + PX_M * 0.24], fill=col)

    ag, am = e.get('ag'), e.get('am')
    if ag is not None and am is not None:
        for j in range(min(len(ag), len(am))):
            if am[j] < 0.5:
                break
            px, py = to_px(float(ag[j][0]), float(ag[j][1]))
            d.rectangle([px - 4, py - 3, px + 4, py + 3], fill=(150, 150, 160))

    route = e.get('route')
    if route is not None and len(route) > 1:
        pts = [to_px(float(r[0]), float(r[1])) for r in route]
        for i in range(0, len(pts) - 1, 2):
            d.line([pts[i], pts[i + 1]], fill=(70, 130, 240), width=2)

    modes = e.get('modes')
    if modes is not None:
        pal = [(120, 120, 255), (120, 200, 255), (150, 255, 150),
               (255, 230, 120), (230, 150, 255), (255, 170, 120)]
        for m in range(len(modes)):
            pts = [to_px(0, 0)] + [to_px(float(p[0]), float(p[1])) for p in modes[m]]
            d.line(pts, fill=pal[m % len(pal)], width=1)

    pred = e.get('pred')
    if pred is not None:
        pts = [to_px(0, 0)] + [to_px(float(p[0]), float(p[1])) for p in pred]
        d.line(pts, fill=(235, 50, 50), width=3)

    ex, ey = to_px(0, 0)
    d.polygon([(ex, ey - 9), (ex - 6, ey + 7), (ex + 6, ey + 7)], fill=(250, 250, 250))

    pose = e.get('pose', np.zeros(4))
    ctrl = e.get('ctrl', np.zeros(3))
    k = int(e.get('k', 0))
    txt = [f'v={float(pose[3]):5.2f} m/s    mode={k}',
           f'thr={float(ctrl[1]):.2f} brk={float(ctrl[2]):.2f} steer={float(ctrl[0]):+.3f}']
    for i, t in enumerate(txt):
        d.text((352, 12 + 18 * i), t, fill=(235, 235, 235))
    leg = [((235, 50, 50), 'selected'), ((120, 120, 255), '6 modes'),
           ((70, 130, 240), 'route'), ((210, 55, 55), 'static occ'),
           ((65, 65, 85), 'vehicles'), ((150, 150, 160), 'agents')]
    for i, (c, name) in enumerate(leg):
        y = 66 + 17 * i
        d.rectangle([352, y, 362, y + 10], fill=c)
        d.text((368, y - 2), name, fill=(190, 190, 190))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--dbg', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--every', type=int, default=2)
    ap.add_argument('--max', type=int, default=150)
    a = ap.parse_args()

    z = np.load(a.dbg, allow_pickle=True)
    keys = z.files
    n = len(z[keys[0]])
    print('dbg 键:', keys, '帧数', n)

    fs = sorted(glob.glob(os.path.join(a.video, 'frame_*.png')))[::a.every][:a.max]
    out = []
    for f in fs:
        step = int(re.search(r'frame_(\d+)', f).group(1))
        i = min(max(step - 1, 0), n - 1)
        e = {k: z[k][i] for k in keys}
        cam = Image.open(f).convert('RGB').resize((640, 360))
        canvas = Image.new('RGB', (1280, 360), (0, 0, 0))
        canvas.paste(cam, (0, 0))
        panel = Image.new('RGB', (PANEL_W, PANEL_H), (16, 16, 20))
        draw_panel(ImageDraw.Draw(panel), e)
        canvas.paste(panel, (640, 0))
        out.append(canvas)

    if not out:
        print('没有帧')
        return
    out[0].save(a.out, save_all=True, append_images=out[1:], duration=120, loop=0)
    print(f'保存 {a.out}：{len(out)} 帧 {out[0].size}')


if __name__ == '__main__':
    main()
