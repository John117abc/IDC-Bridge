"""M2 正式训练：Plant2 向量数据，K=6 自由输出 + WTA，40 epoch。

- 输出无缓冲（python -u 启动），便于实时监控
- 每 epoch 存 checkpoint（按 val minFDE 最优）
- 显存保护：bs=64，OOM 时自动降到 32 重试
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, '/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/tools')
from train_plant2_small import load, add_route_tokens   # noqa: E402
from train_planner import Planner                      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/mnt/2T_HDD/plant2_vector/vector_ds_full')
    ap.add_argument('--out', default='/mnt/2T_HDD/plant2_vector/ckpt')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--k', type=int, default=6)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(args.data, '*.npz')))
    print(f'数据文件 {len(paths)} 个', flush=True)
    E, A, AM, M, MM, R, G = load(paths)
    MP, MM2 = add_route_tokens(M, MM, R)
    del M, MM
    print(f'样本 {len(G)}  (map token {MP.shape[1]})', flush=True)

    n = len(G)
    idx = np.random.RandomState(0).permutation(n)
    n_val = max(256, n // 10)
    va, tr = idx[:n_val], idx[n_val:]
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'训练 {len(tr)} 验证 {len(va)} dev={dev}', flush=True)

    model = Planner(K=args.k).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    def batch(ids):
        return (torch.tensor(E[ids], device=dev),
                torch.tensor(A[ids], device=dev),
                torch.tensor(AM[ids], device=dev),
                torch.tensor(MP[ids], device=dev),
                torch.tensor(MM2[ids], device=dev),
                torch.tensor(G[ids], device=dev))

    @torch.no_grad()
    def evaluate(ids):
        model.eval()
        ade, fde, div = [], [], []
        for i in range(0, len(ids), 256):
            b = batch(ids[i:i + 256])
            pred = model(*b[:5])
            err = (pred - b[5][:, None]).abs()
            ade.append(err.mean(dim=(2, 3)).min(1)[0])
            fde.append(err[:, :, -1].mean(2).min(1)[0])
            if pred.shape[1] > 1:
                ep = pred[:, :, -1]
                d = torch.cdist(ep, ep)
                k = pred.shape[1]
                mask = ~torch.eye(k, dtype=torch.bool, device=ep.device)[None]
                div.append(d[mask.expand_as(d)].reshape(len(ep), -1).min(1)[0])
        model.train()
        return (float(torch.cat(ade).mean()), float(torch.cat(fde).mean()),
                float(torch.cat(div).mean()) if div else 0.0)

    best = 1e9
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        perm = np.random.RandomState(ep).permutation(tr)
        tot, cnt, bs = 0.0, 0, args.bs
        i = 0
        while i < len(perm) - bs + 1:
            ids = perm[i:i + bs]
            try:
                b = batch(ids)
                pred = model(*b[:5])
                err = (pred - b[5][:, None]).abs().mean(dim=(2, 3))
                loss = err.min(1)[0].mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            except torch.cuda.OutOfMemoryError:
                print(f'  OOM @bs={bs} -> 降到 {bs // 2}', flush=True)
                torch.cuda.empty_cache()
                bs = max(8, bs // 2)
                continue
            tot += float(loss); cnt += 1
            i += bs
        sched.step()
        mae, mfde, mdiv = evaluate(va)
        dt = time.time() - t0
        print(f'ep {ep:2d}/{args.epochs} wta {tot/max(cnt,1):.3f} | '
              f'val minADE {mae:.3f} minFDE {mfde:.3f} minPairDiv {mdiv:.2f} | '
              f'bs {bs} {dt:.0f}s', flush=True)
        ck = dict(model=model.state_dict(), K=args.k, epoch=ep,
                  val_minFDE=mfde, args=vars(args))
        torch.save(ck, os.path.join(args.out, 'latest.pt'))
        if mfde < best:
            best = mfde
            torch.save(ck, os.path.join(args.out, 'best.pt'))
        print(f'  saved latest.pt (best {best:.3f})', flush=True)
    print('训练完成', flush=True)


if __name__ == '__main__':
    main()
