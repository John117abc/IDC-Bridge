"""分析 #1 横向偏差 与 #2 规划线长度。"""
import glob
import os

import numpy as np

D = '/mnt/2T_HDD/plant2_vector'
fs = sorted(glob.glob(os.path.join(D, 'dbg_*.npz')), key=os.path.getsize)
Z = np.load(fs[-1])
route, pred, pose, ctrl = Z['route'], Z['pred'], Z['pose'], Z['ctrl']
N = len(route)
print('dump', fs[-1], 'frames', N)

# ---------- #1 横向偏差 ----------
r0 = []          # route 最近点在 ego 系的横向偏移
plan_y_at10 = [] # plan 在 x=10m 处的横向
route_y_at10 = []
plan_len = []
route_len = []
for i in range(N):
    r = route[i]
    rv = np.abs(r[:, 0]) + np.abs(r[:, 1]) > 0
    if rv.sum() < 2:
        continue
    R = r[rv]
    r0.append((R[0, 0], R[0, 1]))
    # 在 x=10 处插值比较
    if R[-1, 0] > 10.0 and R[0, 0] < 10.0:
        order = np.argsort(R[:, 0])
        route_y_at10.append(float(np.interp(10.0, R[order, 0], R[order, 1])))
        P = np.vstack([[0.0, 0.0], pred[i]])
        o2 = np.argsort(P[:, 0])
        plan_y_at10.append(float(np.interp(10.0, P[o2, 0], P[o2, 1])))
    plan_len.append(float(np.linalg.norm(pred[i][-1])))
    route_len.append(float(R[-1, 2]) if R.shape[1] > 2 else np.nan)

r0 = np.array(r0)
print('\n=== #1 横向偏差 ===')
print('route 最近点(ego系): x 中位 %.2f  y 中位 %.2f  |y| 中位 %.2f  p90 %.2f'
      % (np.median(r0[:, 0]), np.median(r0[:, 1]),
         np.median(np.abs(r0[:, 1])), np.percentile(np.abs(r0[:, 1]), 90)))
if plan_y_at10:
    a = np.array(plan_y_at10); b = np.array(route_y_at10)
    print('x=10m 处: route y 中位 %.2f  plan y 中位 %.2f  差(plan-route) 中位 %.2f'
          % (np.median(b), np.median(a), np.median(a - b)))
    print('           |plan-route| 中位 %.2f  p90 %.2f'
          % (np.median(np.abs(a - b)), np.percentile(np.abs(a - b), 90)))

print('\n=== #2 规划线长度 ===')
print('plan 末点距离: 中位 %.1f m  p10 %.1f  p90 %.1f'
      % (np.median(plan_len), np.percentile(plan_len, 10),
         np.percentile(plan_len, 90)))
rl = np.array([x for x in route_len if not np.isnan(x)])
print('route 可见长度: 中位 %.1f m  p10 %.1f  p90 %.1f' %
      (np.median(rl), np.percentile(rl, 10), np.percentile(rl, 90)))
print('\n=== 车速/控制 ===')
print('v 中位 %.2f  最大 %.2f' % (np.median(pose[:, 3]), pose[:, 3].max()))
print('throttle 中位 %.2f   brake 中位 %.2f' % (np.median(ctrl[:, 1]), np.median(ctrl[:, 2])))
