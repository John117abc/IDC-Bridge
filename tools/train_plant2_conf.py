"""M2b：加置信度头重训（WTA L1 + CE(赢家)），输出 top-1 指标。"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/tools')
from train_plant2_small import load, add_route_tokens   # noqa: E402
from train_planner import Planner                      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/mnt/2T_HDD/plant2_vector/vector_ds_full')
    ap.add_argument('--out', default='/mnt/2T_HDD/plant2_vector/ckpt_conf')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--lam-ce', type=float, default=0.5)
    ap.add_argument('--k', type=int, default=6)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(args.data, '*.npz')))
    E, A, AM, M, MM, R, G = load(paths)
    MP, MM2 = add_route_tokens(M, MM, R)
    del M, MM
    n = len(G)
    idx = np.random.RandomState(0).permutation(n)
    n_val = max(256, n // 10)
    va, tr = idx[:n_val], idx[n_val:]
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'样本 {n} 训练 {len(tr)} 验证 {len(va)} dev={dev}', flush=True)

    model = Planner(K=args.k).to(dev)
    model.emit_logits = True
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    def batch(ids):
        return (torch.tensor(E[ids], device=dev), torch.tensor(A[ids], device=dev),
                torch.tensor(AM[ids], device=dev), torch.tensor(MP[ids], device=dev),
                torch.tensor(MM2[ids], device=dev), torch.tensor(G[ids], device=dev))

    @torch.no_grad()
    def evaluate(ids):
        model.eval()
        ade_min, fde_min, fde_top1, acc = [], [], [], []
        for i in range(0, len(ids), 256):
            b = batch(ids[i:i + 256])
            pred, logits = model(*b[:5])
            err = (pred - b[5][:, None]).abs()
            e_ade = err.mean(-1)                      # (B,K)
            e_fde = err[..., -1].mean(-1)
            ade_min.append(e_ade.min(1)[0]); fde_min.append(e_fde.min(1)[0])
            pick = logits.argmax(1)
            fde_top1.append(e_fde.gather(1, pick[:, None])[:, 0])
            acc.append((pick == e_fde.argmin(1)).float())
        model.train()
        return (float(torch.cat(ade_min).mean()), float(torch.cat(fde_min).mean()),
                float(torch.cat(fde_top1).mean()), float(torch.cat(acc).mean()))

    best = 1e9
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        perm = np.random.RandomState(ep).permutation(tr)
        tot, totc, cnt, bs = 0.0, 0.0, 0, args.bs
        i = 0
        while i < len(perm) - bs + 1:
            ids = perm[i:i + bs]
            try:
                b = batch(ids)
                pred, logits = model(*b[:5])
                err = (pred - b[5][:, None]).abs().mean(dim=(2, 3))   # (B,K)
                win = err.argmin(1)
                loss_wta = err.gather(1, win[:, None])[:, 0].mean()
                loss_ce = F.cross_entropy(logits, win)
                loss = loss_wta + args.lam_ce * loss_ce
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            except torch.cuda.OutOfMemoryError:
                print(f'  OOM -> bs {bs}->{bs//2}', flush=True)
                torch.cuda.empty_cache(); bs = max(8, bs // 2); continue
            tot += float(loss_wta); totc += float(loss_ce); cnt += 1
            i += bs
        sched.step()
        mae, mfde, f1, acc = evaluate(va)
        print(f'ep {ep:2d}/{args.epochs} wta {tot/max(cnt,1):.3f} ce {totc/max(cnt,1):.3f} | '
              f'val minADE {mae:.3f} minFDE {mfde:.3f} | top1FDE {f1:.3f} acc {acc*100:.1f}% | '
              f'{time.time()-t0:.0f}s', flush=True)
        ck = dict(model=model.state_dict(), K=args.k, epoch=ep, val_minFDE=mfde,
                  val_top1FDE=f1, val_acc=acc, args=vars(args))
        torch.save(ck, os.path.join(args.out, 'latest.pt'))
        if mfde < best:
            best = mfde
            torch.save(ck, os.path.join(args.out, 'best.pt'))
    print('训练完成', flush=True)


if __name__ == '__main__':
    main()
