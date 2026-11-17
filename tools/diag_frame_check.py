"""检查坐标变换：专家未来 / 锚点 ego 系 / 锚点转世界后是否合理。"""
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "tools"))

from metrics.openloop_pdms import Scene  # noqa: E402
from build_head_dataset import build_one, to_ego  # noqa: E402
from eval_head_pdms import densify, ego_to_world  # noqa: E402

D = '/workspace/data/head_dd/medoid'
JSON_DIR = '/data/data_json/training'
names = ['tfrecord-00000-of-01000_112.json',
         'tfrecord-00000-of-01000_135.json',
         'tfrecord-00000-of-01000_108.json']

import torch  # noqa: E402
from train_head import Head  # noqa: E402

A = np.load(os.path.join(D, 'anchors.npy'))
ck = torch.load(os.path.join(D, 'head_best.pt'), map_location='cpu')
model = Head(ck.get('K', A.shape[0]), ck.get('T', A.shape[1]))
model.load_state_dict(ck['model'])
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(dev).eval()
anchors = torch.tensor(A, dtype=torch.float32, device=dev)
z = np.load(os.path.join(D, 'train.npz'))
ego_m, ego_s = z['ego'].mean(0), z['ego'].std(0) + 1e-3
am = z['amask'].astype(bool)
ag_m, ag_s = z['agent'][am].mean(0), z['agent'][am].std(0) + 1e-3
mm = z['mmask'].astype(bool)
mp_m, mp_s = z['map'][mm].mean(0), z['map'][mm].std(0) + 1e-3

print('=== 锚点原始形状（ego 系，最后一帧）===')
for k in range(len(A)):
    print(f'  anchor{k:2d}: 终点 ({A[k,-1,0]:+7.1f}, {A[k,-1,1]:+7.1f})')

for n in names:
    d = json.load(open(os.path.join(JSON_DIR, n)))
    sc = Scene(d, horizon=40)
    s = build_one(d, A)
    p0, h0 = sc.expert_pos[sc.t_now], float(sc.expert_heading[sc.t_now])
    fut = sc.expert_future[:40]
    fut_ego = to_ego(fut, p0, h0)
    print(f'\n=== {n} ===')
    print(f'  自车 frame10 位置 ({p0[0]:.1f},{p0[1]:.1f}) 朝向 {h0:+.2f} rad')
    print(f'  专家未来 世界系 终点 ({fut[-1,0]:.1f},{fut[-1,1]:.1f})'
          f'  相对位移 ({fut[-1,0]-p0[0]:+.1f},{fut[-1,1]-p0[1]:+.1f})')
    print(f'  专家未来 ego系 终点 ({fut_ego[-1,0]:+.1f},{fut_ego[-1,1]:+.1f})'
          f'  ← 应为"前方(x>0)"')
    # 往返一致性
    rt = ego_to_world(fut_ego, p0, h0)
    print(f'  往返误差 {np.abs(rt-fut).max():.2e}')
    # 头的 top-5
    with torch.no_grad():
        ego = torch.tensor((s['ego'] - ego_m) / ego_s, device=dev)[None]
        ag = torch.tensor((s['agent'] - ag_m) / ag_s, device=dev)[None]
        am_ = torch.tensor(s['amask'], device=dev)[None]
        mp_ = torch.tensor((s['map'] - mp_m) / mp_s, device=dev)[None]
        mm_ = torch.tensor(s['mmask'], device=dev)[None]
        logits, _ = model(ego, ag, am_, mp_, mm_, anchors)
        prob = torch.softmax(logits, -1)[0].cpu().numpy()
    order = np.argsort(-prob)[:5]
    print('  头 top-5:')
    for k in order:
        w = ego_to_world(densify(A[k], 40), p0, h0)
        seg = w[1] - w[0]
        bear = np.arctan2(seg[1], seg[0])
        dbear = np.arctan2(np.sin(bear - h0), np.cos(bear - h0))
        print(f'    k={k:2d} p={prob[k]:.2f} | ego终点'
              f'({A[k,-1,0]:+6.1f},{A[k,-1,1]:+6.1f}) | 世界起点'
              f'({w[0,0]:.1f},{w[0,1]:.1f}) 起点误差'
              f'{np.linalg.norm(w[0]-p0):.2f}m | 首段朝向偏差 '
              f'{np.degrees(dbear):+.1f}°')
