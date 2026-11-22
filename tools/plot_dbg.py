"""诊断：闭环里 route 特征 vs 模型输出 vs 实际轨迹（全部转到 ego 系对比）。"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

Z = np.load('/mnt/2T_HDD/plant2_vector/dbg.npz')
route, pred, pose, ctrl = Z['route'], Z['pred'], Z['pose'], Z['ctrl']
print('frames', len(route))
print('ctrl steer abs mean %.4f  max %.3f' % (np.abs(ctrl[:, 0]).mean(), np.abs(ctrl[:, 0]).max()))
print('thr mean %.2f  brk mean %.2f' % (ctrl[:, 1].mean(), ctrl[:, 2].mean()))
print('route y abs mean %.2f  pred y abs mean %.2f' %
      (np.abs(route[..., 1]).mean(), np.abs(pred[..., 1]).mean()))
print('route 有效行数 mean', float((np.abs(route[..., 0]) > 1e-3).sum(1).mean()))
print('pred 末点 x mean %.2f  y mean %.2f' % (pred[:, -1, 0].mean(), pred[:, -1, 1].mean()))

# 选几帧画图：route（灰点线）、pred（红实线）、实际轨迹（蓝，从 pose 反算到该帧 ego 系）
idx = np.linspace(20, len(route) - 20, 4).astype(int)
fig, axes = plt.subplots(1, len(idx), figsize=(5 * len(idx), 5))
for ax, i in zip(axes, idx):
    x, y, h, _ = pose[i]
    r = route[i]
    m = np.abs(r[:, 0]) > 1e-3
    ax.plot(r[m, 0], r[m, 1], 'o--', ms=3, lw=1.2, c='#9e9e9e', label='route (input)')
    ax.plot(pred[i][:, 0], pred[i][:, 1], 'o-', ms=4, lw=2.0, c='#d50000', label='pred mode')
    # 实际轨迹（未来 40 帧）转 ego 系
    j2 = min(i + 40, len(pose) - 1)
    c, s = np.cos(-h), np.sin(-h)
    d = pose[i:j2 + 1, :2] - np.array([x, y])
    tr = np.stack([d[:, 0] * c - d[:, 1] * s, d[:, 0] * s + d[:, 1] * c], 1)
    ax.plot(tr[:, 0], tr[:, 1], '-', lw=1.8, c='#2962ff', label='actual (4 s)')
    ax.plot(0, 0, 's', ms=8, c='#2e7d32')
    ax.set_aspect('equal'); ax.grid(alpha=0.3)
    ax.set_title('frame %d  steer=%.3f' % (i, ctrl[i, 0]), fontsize=9)
    ax.set_xlim(-5, 45); ax.set_ylim(-15, 15)
    ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig('/mnt/2T_HDD/plant2_vector/dbg_view.png', dpi=110)
print('saved /mnt/2T_HDD/plant2_vector/dbg_view.png')
