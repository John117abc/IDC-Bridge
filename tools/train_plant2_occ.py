"""M3：给规划器加占据栅格输入后重新训练（WTA + CE，同 train_plant2_conf 口径）

用法：python train_plant2_occ.py --data<vector_ds_occ> --out <ckpt dir> --epochs 40
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

R = '/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge'
sys.path.insert(0, R + '/src')
sys.path.insert(0, R + '/tools')
from train_planner import Planner                      # noqa: E402
from train_plant2_small import add_route_tokens        # noqa: E402


def load_all(paths):
    E, A, AM, M, MM, RR, G, O = [], [], [], [], [], [], [], []
    for p in paths:
        d = np.load(p)
        keys = set(d.files)
        E.append(d['ego']); A.append(d['agent']); AM.append(d['amask'])
        M.append(d['map']); MM.append(d['mmask']); RR.append(d['route'])
        G.append(d['gt'])
        if 'occ' in keys:
            O.append(d['occ'].astype(np.uint8))
        else:
            O.append(np.zeros((len(d['ego']), 3, 64, 64), np.uint8))
    cat = lambda xs: np.concatenate(xs)          # noqa: E731
    return (cat(E), cat(A), cat(AM), cat(M), cat(MM), cat(RR), cat(G), cat(O))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--occ-aux", action="store_true",
                    help="占据栅格 aux 重建监督")
    ap.add_argument("--lam-occ", type=float, default=0.1)
    ap.add_argument("--aux-pos-weight", type=float, default=20.0,
                    help="占据为正类的 BCE pos_weight（稀疏补偿）")
    ap.add_argument('--data', default='/mnt/2T_HDD/plant2_vector/vector_ds_occ')
    ap.add_argument('--out', default='/mnt/2T_HDD/plant2_vector/ckpt_occ')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--bs', type=int, default=48)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--lam-ce', type=float, default=0.5)
    ap.add_argument('--k', type=int, default=6)
    ap.add_argument('--max-samples', type=int, default=0,
                    help='0=全部；否则按 --seed 抽样该数量（A/B 必须一致）')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--patience', type=int, default=5,
                    help='验证集连续多少个 epoch 不改善就早停（0=关闭）')
    ap.add_argument('--occ-pos', action='store_true',
                    help='给 occ token 加自车系 (x,y) 位置编码')
    ap.add_argument('--no-occ', action='store_true',
                    help='关掉占据栅格输入（训练对照基线）')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(args.data, '*.npz')))
    print(f'数据集 {len(paths)} 个文件', flush=True)
    E, A, AM, M, MM, RR, G, O = load_all(paths)
    MP, MM2 = add_route_tokens(M, MM, RR)
    del M, MM
    n = len(G)
    if args.max_samples and args.max_samples < n:
        keep = np.random.RandomState(args.seed).choice(
            n, args.max_samples, replace=False)
        keep.sort()
        E, A, AM, MP, MM2, G, O = (E[keep], A[keep], AM[keep],
                                   MP[keep], MM2[keep], G[keep], O[keep])
        n = len(G)
        print(f'抽样到 {n} 样本（seed={args.seed}）', flush=True)
    idx = np.random.RandomState(0).permutation(n)
    n_val = max(256, n // 10)
    va, tr = idx[:n_val], idx[n_val:]
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'样本 {n} 训练 {len(tr)} 验证 {len(va)} dev={dev} '
          f'occ 非零比例 {(O.reshape(n, -1).sum(1) > 0).mean():.2%}', flush=True)

    use_occ = not args.no_occ
    model = Planner(K=args.k, use_occ=use_occ,
                    occ_pos=bool(args.occ_pos), occ_aux=bool(args.occ_aux)).to(dev)
    model.emit_logits = True
    # 从旧 checkpoint 热启动（去掉 occ 相关层以外都能对上的部分）
    old = os.environ.get('VEC_INIT_CKPT',
                         '/mnt/2T_HDD/plant2_vector/plant2_conf.pt')
    if os.path.exists(old):
        try:
            ck = torch.load(old, map_location='cpu', weights_only=False)
            sd = ck['model']
            own = model.state_dict()
            ok = {k: v for k, v in sd.items()
                  if k in own and own[k].shape == v.shape}
            model.load_state_dict(ok, strict=False)
            print(f'热启动 {len(ok)}/{len(own)} 个张量', flush=True)
        except Exception as e:                       # noqa: BLE001
            print('热启动失败', e, flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    def batch(ids):
        return (torch.tensor(E[ids], device=dev),
                torch.tensor(A[ids], device=dev),
                torch.tensor(AM[ids], device=dev),
                torch.tensor(MP[ids], device=dev),
                torch.tensor(MM2[ids], device=dev),
                torch.tensor(O[ids], device=dev).float(),
                torch.tensor(G[ids], device=dev))

    @torch.no_grad()
    def evaluate(ids):
        model.eval()
        a_, f_, f1_, ac_ = [], [], [], []
        for i in range(0, len(ids), 128):
            b = batch(ids[i:i + 128])
            pred, logits = (model(*b[:5], occ=b[5]) if use_occ
                            else model(*b[:5]))
            err = (pred - b[6][:, None]).abs()
            e_ade = err.mean(-1)
            e_fde = err[..., -1].mean(-1)
            a_.append(e_ade.min(1)[0]); f_.append(e_fde.min(1)[0])
            pick = logits.argmax(1)
            f1_.append(e_fde.gather(1, pick[:, None])[:, 0])
            ac_.append((pick == e_fde.argmin(1)).float())
        model.train()
        return (float(torch.cat(a_).mean()), float(torch.cat(f_).mean()),
                float(torch.cat(f1_).mean()), float(torch.cat(ac_).mean()))

    best = 1e9
    no_improve = 0
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        perm = np.random.RandomState(ep).permutation(len(tr))
        tot = totc = 0.0
        cnt = 0
        i = 0
        bs = args.bs
        while i < len(perm) - bs + 1:
            ids = tr[perm[i:i + bs]]
            try:
                b = batch(ids)
                pred, logits = (model(*b[:5], occ=b[5]) if use_occ
                                else model(*b[:5]))
                err = (pred - b[6][:, None]).abs().mean(dim=(2, 3))
                win = err.argmin(1)
                loss_wta = err.gather(1, win[:, None])[:, 0].mean()
                loss_ce = F.cross_entropy(logits, win)
                loss = loss_wta + args.lam_ce * loss_ce
                if getattr(args, 'occ_aux', False) and hasattr(model, 'last_aux'):
                    _occ_b = b[5].float()
                    # (B,3,64,64) → 任意通道 → maxpool2 → (B,32,32)
                    _m = (F.max_pool2d(_occ_b, 2).sum(1) > 0).float()   # (B,32,32)
                    # 切成 8x8 个 patch，每个 patch 4x4
                    _tgt = _m.view(-1, 8, 4, 8, 4).permute(0, 1, 3, 2, 4).reshape(-1, 64, 16)
                    _aux = model.last_aux
                    if _aux.shape == _tgt.shape:
                        _pw = torch.tensor(
                            float(getattr(args, 'aux_pos_weight', 20.0)),
                            device=_aux.device)
                        _laux = F.binary_cross_entropy_with_logits(
                            _aux, _tgt, pos_weight=_pw)
                        loss = loss + args.lam_occ * _laux
                        if cnt % 20 == 0:
                            print('    aux %.4f' % float(_laux), flush=True)

                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            except torch.cuda.OutOfMemoryError:
                print(f'  OOM -> bs {bs}->{bs // 2}', flush=True)
                torch.cuda.empty_cache(); bs = max(8, bs // 2); continue
            tot += float(loss_wta); totc += float(loss_ce); cnt += 1
            i += bs
        sched.step()
        mae, mfde, f1, acc = evaluate(va)
        print(f'ep {ep:2d}/{args.epochs} wta {tot/max(cnt,1):.3f} ce {totc/max(cnt,1):.3f} | '
              f'val minADE {mae:.3f} minFDE {mfde:.3f} | top1FDE {f1:.3f} '
              f'acc {acc*100:.1f}% | {time.time()-t0:.0f}s', flush=True)
        ck = dict(model=model.state_dict(), K=args.k, epoch=ep, use_occ=use_occ,
                  occ_pos=bool(args.occ_pos),
                  val_minFDE=mfde, val_top1FDE=f1, val_acc=acc, args=vars(args))
        torch.save(ck, os.path.join(args.out, 'latest.pt'))
        if mfde < best - 1e-4:
            best = mfde
            no_improve = 0
            torch.save(ck, os.path.join(args.out, 'best.pt'))
            print(f'  ↑ best {mfde:.3f}', flush=True)
        else:
            no_improve += 1
            print(f'  · 未改善 {no_improve}/{args.patience}', flush=True)
            if args.patience > 0 and no_improve >= args.patience:
                print(f'早停：连续 {no_improve} 个 epoch 未改善', flush=True)
                break
    print('训练完成 best minFDE', round(best, 3), flush=True)


if __name__ == '__main__':
    main()
