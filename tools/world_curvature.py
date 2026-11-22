"""列出 seed 下每个 world 的场景名与转向量（用于挑"转弯世界"）。

用法： python tools/world_curvature.py --num-worlds 12 --seed 42
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, '/gpudrive/baselines/IDC-Bridge/src')
sys.path.insert(0, '/gpudrive/baselines/IDC-Bridge/tools')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='/data/data_json/training')
    ap.add_argument('--num-worlds', type=int, default=12)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    from gpudrive.env.dataset import SceneDataLoader
    loader = SceneDataLoader(root=args.data_dir,
                             batch_size=args.num_worlds,
                             dataset_size=args.num_worlds,
                             sample_with_replacement=True,
                             shuffle=True, seed=args.seed)
    idx = list(loader.indices)[:args.num_worlds]
    files = [loader.dataset[i] for i in idx]
    print('world  |dH|度  弧长m  文件名')
    rows = []
    for w, f in enumerate(files):
        d = json.load(open(f))
        o = d['objects'][int(d['metadata']['sdc_track_index'])]
        P = np.asarray([[q['x'], q['y']] for q in o['position']], np.float64)
        H = np.asarray(o.get('heading') or [0.0] * len(P), np.float64)
        t = 10
        seg = np.diff(P[t:], axis=0)
        L = np.linalg.norm(seg, axis=1)
        arc = float(L.sum())
        dh = 0.0
        for i in range(t, len(H) - 1):
            dh += abs(float(np.arctan2(np.sin(H[i + 1] - H[i]),
                                       np.cos(H[i + 1] - H[i]))))
        rows.append((w, np.degrees(dh), arc, os.path.basename(f)))
        print('%5d | %6.1f | %6.1f | %s' % (w, np.degrees(dh), arc,
                                            os.path.basename(f)))
    best = max(rows, key=lambda r: r[1])
    print('\n转向最多: world %d（%.0f 度，弧长 %.0f m）' % (best[0], best[1], best[2]))
    moving = [r for r in rows if r[2] > 20.0]
    if moving:
        b2 = max(moving, key=lambda r: r[1])
        print('会动且转向最多: world %d（%.0f 度，弧长 %.0f m）'
              % (b2[0], b2[1], b2[2]))


if __name__ == '__main__':
    main()
