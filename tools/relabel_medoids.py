"""把锚点从"聚类均值"换成"medoid（真实 GT 轨迹）"，并重新打标签。

复用已有 train.npz 里的 gt，不需要重读 JSON。

    python tools/relabel_medoids.py --data /workspace/data/head60 --k 16
"""
import argparse
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "tools"))

from build_head_dataset import kmeans  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    z = np.load(os.path.join(args.data, 'train.npz'))
    gt = z['gt']                                   # (N,T,2)
    N = len(gt)
    X = gt.reshape(N, -1).astype(np.float32)
    print(f'样本 {N}  轨迹维度 {X.shape[1]}')

    C, lab = kmeans(X, args.k, seed=args.seed)     # lab: (N,)
    anchors = np.zeros((args.k,) + gt.shape[1:], np.float32)
    for i in range(args.k):
        m = np.where(lab == i)[0]
        if len(m) == 0:
            continue
        d = ((X[m] - C[i]) ** 2).sum(1)
        anchors[i] = gt[m[int(np.argmin(d))]]       # medoid：类内最像中心的**真实轨迹**
    print(f'锚点 {anchors.shape}（medoid）')

    # 重新打标签：按 FDE 最近的锚点
    fde = np.linalg.norm(anchors[:, None, -1, :] - gt[None, :, -1, :], axis=-1)  # (K,N)
    label = fde.argmin(0)
    resid = gt - anchors[label]
    best = fde.min(0)
    print(f'标签 FDE: p50={np.median(best):.2f} p90={np.percentile(best,90):.2f} '
          f'<1m {100*np.mean(best<1):.0f}%')

    out_dir = os.path.join(args.data, f'medoid{args.k}')
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'anchors.npy'), anchors)
    d = {k: z[k] for k in z.files if k not in ('label', 'resid')}
    d['label'] = label.astype(np.int64)
    d['resid'] = resid.astype(np.float32)
    np.savez_compressed(os.path.join(out_dir, 'train.npz'), **d)
    print('saved to', out_dir)
    print('关键对比：均值锚点 p50=4.50（已知），medoid 见上')


if __name__ == '__main__':
    main()
