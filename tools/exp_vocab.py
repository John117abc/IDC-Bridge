"""词汇表规模实验：K = 64/256/1024 时，oracle FDE/ADE 能到多少。

用已有 train.npz 里的 gt（自车系 8 waypoint × 0.5 s）聚类，评估覆盖率上限。
"""
import os
import sys

import numpy as np

D = '/workspace/data/head_dd/medoid'
KS = [20, 64, 256, 1024]
SUB = 20000          # 聚类用的子样本
CHUNK = 4000


def kmeans_chunked(X, k, iters=15, seed=0):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    C = np.empty((k, d), np.float32)
    C[0] = X[rng.integers(n)]
    d2 = ((X - C[0]) ** 2).sum(1)
    for i in range(1, k):
        p = d2 / max(d2.sum(), 1e-9)
        C[i] = X[rng.choice(n, p=p)]
        d2 = np.minimum(d2, ((X - C[i]) ** 2).sum(1))
    for it in range(iters):
        lab = np.empty(n, np.int32)
        for s in range(0, n, CHUNK):
            e = min(s + CHUNK, n)
            dist = ((X[s:e, None, :] - C[None]) ** 2).sum(-1)
            lab[s:e] = dist.argmin(1)
        for i in range(k):
            m = lab == i
            if m.any():
                C[i] = X[m].mean(0)
        if it % 5 == 4:
            print(f'    kmeans iter {it+1}/{iters}', flush=True)
    return C


def main():
    z = np.load(os.path.join(D, 'train.npz'))
    gt = z['gt'].astype(np.float32)            # (N, 8, 2)
    N = len(gt)
    X = gt.reshape(N, -1)
    print(f'样本 {N}  轨迹维度 {X.shape[1]}')

    # 专家未来本身的横向偏移（判断"转弯场景"占比）
    lat = np.abs(gt[:, :, 1]).max(1)
    print(f'专家横向偏移: p50={np.median(lat):.2f}  '
          f'>4m 占比 {100*np.mean(lat > 4):.1f}%')

    rng = np.random.default_rng(0)
    sub = rng.choice(N, min(SUB, N), replace=False)

    print(f"\n{'K':>6s} {'oracle FDE p50':>14s} {'FDE p90':>8s} "
          f"{'oracle ADE p50':>15s} {'ADE p90':>8s} {'FDE<1m':>7s} "
          f"{'转弯锚点':>7s}")
    for K in KS:
        C = kmeans_chunked(X[sub], K, seed=0)
        fde = np.empty(N, np.float32)
        ade = np.empty(N, np.float32)
        best = np.empty(N, np.int32)
        for s in range(0, N, CHUNK):
            e = min(s + CHUNK, N)
            diff = X[s:e, None, :] - C[None]          # (B,K,D)
            d2 = (diff ** 2).sum(-1)
            j = d2.argmin(1)
            best[s:e] = j
            A = C.reshape(K, 8, 2)
            xe = (X[s:e].reshape(-1, 8, 2) - A[j])    # (B,8,2)
            fde[s:e] = np.linalg.norm(xe[:, -1, :], axis=1)
            ade[s:e] = np.linalg.norm(xe, axis=2).mean(1)
        A = C.reshape(K, 8, 2)
        turn = np.abs(A[:, :, 1]).max(1) > 4
        print(f'{K:6d} {np.median(fde):14.2f} {np.percentile(fde, 90):8.2f} '
              f'{np.median(ade):15.2f} {np.percentile(ade, 90):8.2f} '
              f'{100*np.mean(fde < 1):6.0f}% {int(turn.sum()):5d}/{K}')


if __name__ == '__main__':
    main()
