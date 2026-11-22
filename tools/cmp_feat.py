"""P0：离线（JSON 组装）vs 在线（env 张量组装）的规划器特征逐块对比。

用法：
    python tools/cmp_feat.py --dump /workspace/data/rt_feat.npz --n 4
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, '/gpudrive/baselines/IDC-Bridge/src')
sys.path.insert(0, '/gpudrive/baselines/IDC-Bridge/tools')

from build_head_dataset import build_one  # noqa: E402

EGO_NAMES = ['v_now', 'v_prev', 'yaw_rate'] + [
    'hist%d_%s' % (i // 2, 'xy'[i % 2]) for i in range(20)]
AG_NAMES = ['rel_x', 'rel_y', 'cos_dh', 'sin_dh', 'v', 'len', 'width', 'valid'] + [
    'hist%d_%s' % (i // 2, 'xy'[i % 2]) for i in range(20)]
MP_NAMES = ['mid_x', 'mid_y', 'len', 'cos', 'sin', 'is_lane', 'valid']


def rel_diff(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(np.abs(a - b).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', default='/workspace/data/rt_feat.npz')
    ap.add_argument('--data-dir', default='/data/data_json/training')
    ap.add_argument('--n', type=int, default=4)
    ap.add_argument('--match-tol', type=float, default=0.5)
    args = ap.parse_args()

    z = np.load(args.dump, allow_pickle=True)
    for w in range(args.n):
        if f'name{w}' not in z.files:
            continue
        name = str(z[f'name{w}'])
        fp = os.path.join(args.data_dir, name)
        if not os.path.exists(fp):
            print(f'world {w}: {name} 不存在，跳过')
            continue
        d = json.load(open(fp))
        smp = build_one(d, np.zeros((6, 8, 2), np.float32))   # t_now = 10
        if smp is None:
            print(f'world {w}: build_one 失败')
            continue
        ego_off, ego_on = smp['ego'], z[f'ego{w}']
        ag_off, ag_on = smp['agent'], z[f'ag{w}']
        am_on = z[f'am{w}']
        mp_off, mp_on = smp['map'], z[f'mp{w}']
        mm_on = z[f'mm{w}']
        pose = z[f'pose{w}']

        print(f'\n===== world {w}  {name}  pose=({pose[0]:.1f},{pose[1]:.1f},{pose[2]:.2f}) =====')
        # ---- ego ----
        de = np.abs(ego_off - ego_on)
        j = int(np.argmax(de))
        print(f'[ego] 形状 {ego_off.shape} vs {ego_on.shape}')
        print(f'  max|Δ|=%.3f  平均|Δ|=%.3f' % (de.max(), de.mean()))
        print(f'  离线前5: {np.round(ego_off[:5], 2)}')
        print(f'  在线前5: {np.round(ego_on[:5], 2)}')
        print(f'  差异最大维: {EGO_NAMES[j]}  离线 %.3f 在线 %.3f'
              % (ego_off[j], ego_on[j]))

        # ---- agent：按位置匹配 ----
        n_on = int((am_on > 0.5).sum())
        n_off = int((smp['amask'] > 0.5).sum())
        print(f'[agent] 离线 {n_off} 个 / 在线 {n_on} 个')
        off_idx = np.where(smp['amask'] > 0.5)[0]
        on_idx = np.where(am_on > 0.5)[0]
        matched, diffs = 0, []
        for oi in on_idx:
            p = ag_on[oi, :2]
            dd = np.linalg.norm(ag_off[off_idx, :2] - p[None], axis=1)
            if len(dd) and dd.min() < args.match_tol:
                matched += 1
                k = off_idx[int(np.argmin(dd))]
                diffs.append(np.abs(ag_off[k] - ag_on[oi]))
        if diffs:
            diffs = np.array(diffs)
            per = diffs.max(0)
            top = np.argsort(-per)[:6]
            print(f'  匹配上 {matched}/{n_on}')
            print(f'  匹配对逐维 max|Δ|: ' +
                  ', '.join('%s=%.2f' % (AG_NAMES[t], per[t]) for t in top))
            print(f'  非历史字段(前8维) max|Δ|=%.2f' % per[:8].max())
            print(f'  历史字段(后20维) max|Δ|=%.2f' % per[8:].max())

        # ---- map ----
        n_off = int((smp['mmask'] > 0.5).sum())
        n_on = int((mm_on > 0.5).sum())
        print(f'[map] 离线 {n_off} 段 / 在线 {n_on} 段')
        mp_off_v = mp_off[smp['mmask'] > 0.5]
        mp_on_v = mp_on[mm_on > 0.5]
        matched, dm = 0, []
        for row in mp_on_v:
            dd = np.linalg.norm(mp_off_v[:, :2] - row[None, :2], axis=1)
            if len(dd) and dd.min() < args.match_tol:
                matched += 1
                k = int(np.argmin(dd))
                dm.append(np.abs(mp_off_v[k] - row))
        if dm:
            dm = np.array(dm)
            per = dm.max(0)
            print(f'  匹配上 {matched}/{n_on}')
            print(f'  逐维 max|Δ|: ' +
                  ', '.join('%s=%.2f' % (MP_NAMES[t], per[t]) for t in range(7)))
            print(f'  类型(is_lane)一致率: %.0f%%'
                  % (100 * np.mean(dm[:, 5] < 0.5)))
            print(f'  长度 max|Δ|=%.2f m' % per[2])


if __name__ == '__main__':
    main()
