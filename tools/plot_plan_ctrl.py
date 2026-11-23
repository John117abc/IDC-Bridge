"""CARLA 闭环：规划线 + 当前控制 的可视化（自车系俯视图）

每帧画：
  · 灰线 = 地图车道段
  · 蓝虚线 = route（参考路线）
  · 红色粗线 = 规划/跟踪轨迹（当前 plan）
  · 橙色框 = 周围车辆
  · 绿方块 = 自车
  · 文本框 = v / steer / throttle / brake
"""
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

D = '/mnt/2T_HDD/plant2_vector'
fs = sorted(glob.glob(os.path.join(D, 'dbg_*.npz')), key=os.path.getsize)
if not fs:
    raise SystemExit('没有 dbg 文件')
src = fs[-1]                     # 用帧数最多的（跑得最久）
Z = np.load(src)
route, pred, pose, ctrl = Z['route'], Z['pred'], Z['pose'], Z['ctrl']
modes, mp, mm = Z['modes'], Z['mp'], Z['mm']
ag = Z['ag'] if 'ag' in Z.files else None
am = Z['am'] if 'am' in Z.files else None
N = len(route)
print('dump', src, 'frames', N)


def draw(i, ax):
    ax.clear()
    m = mm[i] > 0.5
    if m.any():
        mx, my, ln = mp[i, m, 0], mp[i, m, 1], mp[i, m, 2]
        hh = np.arctan2(mp[i, m, 4], mp[i, m, 3])
        dx, dy = np.cos(hh) * ln / 2, np.sin(hh) * ln / 2
        for a, b, u, w in zip(mx - dx, my - dy, mx + dx, my + dy):
            ax.plot([a, u], [b, w], '-', lw=0.9, c='#b0bec5', zorder=1)

    r = route[i]
    rv = np.abs(r[:, 0]) + np.abs(r[:, 1]) > 0
    if rv.sum() >= 2:
        ax.plot(r[rv, 0], r[rv, 1], '--', lw=2.0, c='#1565c0',
                label='route (参考)', zorder=3)

    for k in range(modes.shape[1]):
        P = np.vstack([[0, 0], modes[i, k]])
        ax.plot(P[:, 0], P[:, 1], '-', lw=0.8, c='#90a4ae',
                alpha=0.35, zorder=2)

    P = np.vstack([[0, 0], pred[i]])
    ax.plot(P[:, 0], P[:, 1], '-', lw=3.2, c='#d50000',
            label='plan (规划线)', zorder=6)
    ax.plot(P[-1, 0], P[-1, 1], 'o', ms=7, mfc='none', mec='#d50000',
            mew=1.5, zorder=6)

    if ag is not None:
        for k in range(len(am[i])):
            if am[i][k] < 0.5:
                continue
            px, py = float(ag[i][k, 0]), float(ag[i][k, 1])
            dh = float(np.arctan2(ag[i][k, 3], ag[i][k, 2]))
            L, W = max(float(ag[i][k, 5]), 1.0), max(float(ag[i][k, 6]), 0.5)
            cc, ss = np.cos(dh), np.sin(dh)
            pts = np.array([[-L/2, -W/2], [L/2, -W/2], [L/2, W/2], [-L/2, W/2]])
            Rm = np.array([[cc, -ss], [ss, cc]])
            ax.add_patch(Polygon(pts @ Rm.T + np.array([px, py]), closed=True,
                                 fc='#ffb74d', ec='#e65100', lw=0.7,
                                 alpha=0.9, zorder=5))

    ax.plot(0, 0, 's', ms=11, c='#2e7d32', zorder=7)
    ax.set_xlim(-10, 60)
    ax.set_ylim(-25, 25)
    ax.set_aspect('equal')
    ax.grid(alpha=0.25)
    v = pose[i, 3]
    ax.set_title(f'frame {i}/{N-1}', fontsize=10)
    ax.text(0.02, 0.02,
            f'v = {v:.2f} m/s\nsteer = {ctrl[i,0]:+.2f}\n'
            f'throttle = {ctrl[i,1]:.2f}\nbrake = {ctrl[i,2]:.2f}',
            transform=ax.transAxes, fontsize=10, family='monospace',
            va='bottom', ha='left',
            bbox=dict(fc='white', ec='#455a64', alpha=0.9, boxstyle='round,pad=0.4'))
    if i == 0:
        ax.legend(fontsize=9, loc='upper right')


picks = sorted(set([0, N // 5, 2 * N // 5, 3 * N // 5, N - 1]))
picks = [p for p in picks if 0 <= p < N]
fig, axes = plt.subplots(1, len(picks), figsize=(5.2 * len(picks), 4.6))
if len(picks) == 1:
    axes = [axes]
for ax, i in zip(axes, picks):
    draw(i, ax)
plt.tight_layout()
out = os.path.join(D, 'carla_plan_ctrl.png')
plt.savefig(out, dpi=110)
print('saved', out)
