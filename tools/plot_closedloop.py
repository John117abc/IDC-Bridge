"""闭环俯视图：地图 / route / K 条模态 / 选中模态 / 自车轨迹 → gif。"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image
import io, os

Z = np.load('/mnt/2T_HDD/plant2_vector/dbg.npz')
route, pred, pose, ctrl = Z['route'], Z['pred'], Z['pose'], Z['ctrl']
modes, mp, mm = Z['modes'], Z['mp'], Z['mm']
N = len(route)
print('frames', N, 'modes', modes.shape)

# 世界坐标轨迹（把每帧 ego 位置串起来）
trail = pose[:, :2]

COLS = ['#d50000', '#2962ff', '#00c853', '#ff6d00', '#aa00ff', '#00bcd4']


def draw(i, ax2):
    x, y, h, v = pose[i]
    ax2.clear()
    m = mm[i] > 0.5
    if m.any():
        mx, my, ln, c, s = mp[i, m, 0], mp[i, m, 1], mp[i, m, 2], mp[i, m, 3], mp[i, m, 4]
        hh = np.arctan2(s, c)
        dx, dy = np.cos(hh) * ln / 2, np.sin(hh) * ln / 2
        for a, b, u, w in zip(mx - dx, my - dy, mx + dx, my + dy):
            ax2.plot([a, u], [b, w], '-', lw=0.8, c='#b0bec5', zorder=1)
    r = route[i]
    rv = np.abs(r[:, 0]) + np.abs(r[:, 1]) > 0
    if rv.sum() >= 2:
        ax2.plot(r[rv, 0], r[rv, 1], '--', lw=1.6, c='#1565c0',
                 label='route', zorder=3)
    # K 条模态（ego 系）
    for k in range(modes.shape[1]):
        P = np.vstack([[0, 0], modes[i, k]])
        ax2.plot(P[:, 0], P[:, 1], '-', lw=1.0, c=COLS[k % len(COLS)],
                 alpha=0.45, zorder=4)
    P = np.vstack([[0, 0], pred[i]])
    ax2.plot(P[:, 0], P[:, 1], '-', lw=2.8, c='#d50000', label='selected', zorder=6)
    ax2.plot(0, 0, 's', ms=9, c='#2e7d32', zorder=7)
    ax2.set_xlim(-8, 60)
    ax2.set_ylim(-22, 22)
    ax2.set_aspect('equal')
    ax2.grid(alpha=0.3)
    ax2.set_title('f%03d  v=%.1f  steer=%.2f  thr=%.2f' %
                  (i, v, ctrl[i, 0], ctrl[i, 1]), fontsize=9)
    if i == 0:
        ax2.legend(fontsize=7, loc='upper right')


# 选 route 曲率最大的 4 帧（互不相邻）
curv = np.zeros(N)
for _i in range(N):
    r = route[_i]
    rv = np.abs(r[:, 0]) + np.abs(r[:, 1]) > 0
    if rv.sum() >= 4:
        p = r[rv]
        d = np.diff(p, axis=0)
        th = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        seg = np.maximum(np.linalg.norm(d, axis=1), 1e-3)
        curv[_i] = np.abs(np.gradient(th) / seg).mean()
idxs = []
for _i in np.argsort(-curv):
    if all(abs(int(_i) - _j) > 25 for _j in idxs):
        idxs.append(int(_i))
    if len(idxs) == 4:
        break
print('turn frames', idxs, 'curv', [round(float(curv[i]), 4) for i in idxs])
sp = pose[:, 3]
print('speed mean %.2f  frac<2m/s %.2f' % (sp.mean(), (sp < 2).mean()))
fig4, ax4 = plt.subplots(2, 2, figsize=(13, 9))
for _ax, _i in zip(ax4.ravel(), idxs):
    draw(_i, _ax)
plt.tight_layout()
plt.savefig('/mnt/2T_HDD/plant2_vector/cl_frames.png', dpi=100)
print('saved frames fig')

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6),
                         gridspec_kw={'width_ratios': [1, 1.35]})
frames = []
step = max(1, N // 90)
for i in list(range(0, N, step))[:90]:
    axes[0].clear()
    # 左：世界系（路线 + 自车轨迹）
    axes[0].plot(trail[:i + 1, 0], trail[:i + 1, 1], '-', lw=2.2, c='#2962ff')
    axes[0].plot(trail[i, 0], trail[i, 1], 's', ms=8, c='#2e7d32')
    axes[0].set_aspect('equal')
    _m = 12.0
    axes[0].set_xlim(trail[:, 0].min() - _m, trail[:, 0].max() + _m)
    axes[0].set_ylim(trail[:, 1].min() - _m, trail[:, 1].max() + _m)
    axes[0].set_title('world view (ego trail)', fontsize=9)
    axes[0].grid(alpha=0.3)
    draw(i, axes[1])
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=90)
    buf.seek(0)
    frames.append(Image.open(buf).convert('RGB').copy())
    if len(frames) == 1:
        frames[0].save('/mnt/2T_HDD/plant2_vector/cl_sample.png')
imgs = [im.resize((im.width // 2, im.height // 2)) for im in frames]
imgs[0].save('/mnt/2T_HDD/plant2_vector/cl_view.gif', save_all=True,
             append_images=imgs[1:], duration=140, loop=0)
print('saved gif', len(imgs))
