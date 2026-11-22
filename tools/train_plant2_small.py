"""小训练验证：用 3 个场景的样本，训 K=6 自由输出 + WTA，看 loss 是否下降。

route 以额外 token 的形式并入 map 组（type=2），复用现有 Planner。
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, '/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/tools')
sys.path.insert(0, '/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/src')
from train_planner import Planner      # noqa: E402


def load(paths):
    E, A, AM, M, MM, R, G = [], [], [], [], [], [], []
    for p in paths:
        z = np.load(p)
        E.append(z['ego']); A.append(z['agent']); AM.append(z['amask'])
        M.append(z['map']); MM.append(z['mmask']); R.append(z['route'])
        G.append(z['gt'])
    return (np.concatenate(E), np.concatenate(A), np.concatenate(AM),
            np.concatenate(M), np.concatenate(MM), np.concatenate(R),
            np.concatenate(G))


def add_route_tokens(mp, mm, route):
    """route (B,24,4) -> 作为 map 的额外 token（type=2）。"""
    B = len(mp)
    rp = np.zeros((B, 24, 7), np.float32)
    rp[..., 0] = route[..., 0]
    rp[..., 1] = route[..., 1]
    rp[..., 2] = 2.0
    rp[..., 3] = route[..., 3]
    rp[..., 4] = np.sqrt(np.maximum(1.0 - route[..., 3] ** 2, 0.0))
    rp[..., 5] = 2.0
    rp[..., 6] = (np.abs(route[..., 0]) + np.abs(route[..., 1]) > 0).astype(np.float32)
    rm = rp[..., 6].copy()
    return np.concatenate([mp, rp], 1), np.concatenate([mm, rm], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/mnt/2T_HDD/plant2_vector/vector_ds')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(args.data, '*.npz')))
    paths = [p for p in paths if 'multi_' not in os.path.basename(p)]
    print('数据文件:', [os.path.basename(p) for p in paths])
    E, A, AM, M, MM, R, G = load(paths)
    MP, MM = add_route_tokens(M, MM, R)
    if args.limit:
        E, A, AM, MP, MM, G = (E[:args.limit], A[:args.limit], AM[:args.limit],
                               MP[:args.limit], MM[:args.limit], G[:args.limit])
    n = len(G)
    idx = np.random.RandomState(0).permutation(n)
    n_val = max(64, n // 10)
    tr, va = idx[n_val:], idx[:n_val]
    print(f'样本 {n}  训练 {len(tr)}  验证 {len(va)}')
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Planner(K=6).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    def batch(ids):
        return (torch.tensor(E[ids], device=dev),
                torch.tensor(A[ids], device=dev),
                torch.tensor(AM[ids], device=dev),
                torch.tensor(MP[ids], device=dev),
                torch.tensor(MM[ids], device=dev),
                torch.tensor(G[ids], device=dev))

    @torch.no_grad()
    def evaluate(ids):
        model.eval()
        fde, ade = [], []
        for i in range(0, len(ids), 256):
            b = batch(ids[i:i + 256])
            pred = model(*b[:5])
            err = (pred - b[5][:, None]).abs()
            ade.append(err.mean(dim=(2, 3)).min(1)[0])
            fde.append(err[:, :, -1].mean(2).min(1)[0])
        model.train()
        return float(torch.cat(ade).mean()), float(torch.cat(fde).mean())

    for ep in range(1, args.epochs + 1):
        perm = np.random.RandomState(ep).permutation(tr)
        tot, cnt = 0.0, 0
        for i in range(0, len(perm) - args.bs + 1, args.bs):
            b = batch(perm[i:i + args.bs])
            pred = model(*b[:5])
            err = (pred - b[5][:, None]).abs().mean(dim=(2, 3))   # (B,K)
            loss = err.min(1)[0].mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss); cnt += 1
        mae, mfde = evaluate(va)
        print(f'ep {ep}  train_wta {tot/max(cnt,1):.3f}  '
              f'val minADE {mae:.3f}  val minFDE {mfde:.3f}  (dev={dev})')


if __name__ == '__main__':
    main()
