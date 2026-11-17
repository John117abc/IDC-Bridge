"""规划头：锚点分类 + 残差回归（MTR/WTA 风格，轻量版）。

输入：ego(3) + agents(N,8)+mask + map(M,7)+mask
输出：K 个锚点的 logits + 每个锚点对应的残差 (K,160)
损失：CE + L1(resid[k*]) + L1(anchor[k*]+resid[k*], gt)

用法：
    python tools/train_head.py --data /workspace/data/head --epochs 300
"""
import argparse
import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402


class Head(nn.Module):
    """小 Transformer：agent/map tokens + ego query → 锚点分类 + 残差回归。"""

    def __init__(self, K=16, T=80, d=128, nhead=4, layers=2):
        super().__init__()
        self.K, self.T, self.d = K, T, d
        self.ego_in = nn.Sequential(nn.Linear(23, d), nn.ReLU())
        self.ag_in = nn.Sequential(nn.Linear(28, d), nn.ReLU())
        self.mp_in = nn.Sequential(nn.Linear(7, d), nn.ReLU())
        self.type_emb = nn.Parameter(torch.zeros(2, d))
        enc = nn.TransformerEncoderLayer(d, nhead, 2 * d, dropout=0.0,
                                         batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.cls = nn.Linear(d, K)
        self.anch = nn.Sequential(nn.Linear(T * 2, 128), nn.ReLU(),
                                  nn.Linear(128, 64))
        self.reg = nn.Sequential(nn.Linear(d + 64, 256), nn.ReLU(),
                                 nn.Linear(256, T * 2))

    def forward(self, ego, agent, amask, mp, mmask, anchors, k_hint=None):
        B = ego.shape[0]
        e = self.ego_in(ego)[:, None, :]
        a = self.ag_in(agent) + self.type_emb[0]
        m = self.mp_in(mp) + self.type_emb[1]
        seq = torch.cat([e, a, m], 1)
        pad = torch.cat([torch.ones(B, 1, device=ego.device), amask, mmask], 1) < 0.5
        h = self.enc(seq, src_key_padding_mask=pad)
        q = h[:, 0]
        logits = self.cls(q)
        ae = self.anch(anchors.reshape(self.K, -1))
        k = k_hint if k_hint is not None else logits.argmax(-1)
        resid = self.reg(torch.cat([q, ae[k]], -1))
        resid = torch.clamp(resid, -15.0, 15.0)
        return logits, resid.reshape(-1, self.T, 2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--patience", type=int, default=8, help="早停耐心（每隔 5 轮评估一次）")
    args = ap.parse_args()

    z = np.load(os.path.join(args.data, 'train.npz'))
    anchors = torch.tensor(np.load(os.path.join(args.data, 'anchors.npy')),
                           dtype=torch.float32)
    K, T = anchors.shape[0], anchors.shape[1]
    n = len(z['label'])
    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    nv = max(int(n * args.val_ratio), 1)
    va, tr = idx[:nv], idx[nv:]
    print(f"样本 {n}  训练 {len(tr)}  验证 {len(va)}  锚点 {K}x{T}")

    # ---- 特征标准化（用训练集统计量）----
    tr_ = tr
    ego_m, ego_s = z['ego'][tr_].mean(0), z['ego'][tr_].std(0) + 1e-3
    am = z['amask'][tr_].astype(bool)
    ag_m = z['agent'][tr_][am].mean(0)
    ag_s = z['agent'][tr_][am].std(0) + 1e-3
    mm = z['mmask'][tr_].astype(bool)
    mp_m = z['map'][tr_][mm].mean(0)
    mp_s = z['map'][tr_][mm].std(0) + 1e-3
    print('ego mean/std', np.round(ego_m, 2), np.round(ego_s, 2))

    # ---- 类别权重（抑制多数类塌缩）----
    cnt = np.bincount(z['label'][tr_], minlength=K).astype(np.float64)
    cw = np.where(cnt > 0, (cnt.sum() / (K * np.maximum(cnt, 1))) ** 0.5, 0.0)
    cw = torch.tensor(cw / max(cw.mean(), 1e-6), dtype=torch.float32)
    print('类别权重 min/max', float(cw.min()), float(cw.max()))

    # ---- 一次把数据搬到 GPU（<200 MB），每步只做索引，避免逐批 H2D 拷贝 ----
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    G = {}
    for k in ('ego', 'agent', 'amask', 'map', 'mmask', 'gt', 'label', 'resid'):
        G[k] = torch.as_tensor(z[k], device=dev)
    G['ego_m'] = torch.tensor(ego_m, device=dev).float()
    G['ego_s'] = torch.tensor(ego_s, device=dev).float()
    G['ag_m'] = torch.tensor(ag_m, device=dev).float()
    G['ag_s'] = torch.tensor(ag_s, device=dev).float()
    G['mp_m'] = torch.tensor(mp_m, device=dev).float()
    G['mp_s'] = torch.tensor(mp_s, device=dev).float()
    print('数据已搬到', dev)

    def batch(ix):
        ix_t = torch.as_tensor(ix, device=dev)
        return ((G['ego'][ix_t] - G['ego_m']) / G['ego_s'],
                (G['agent'][ix_t] - G['ag_m']) / G['ag_s'],
                G['amask'][ix_t],
                (G['map'][ix_t] - G['mp_m']) / G['mp_s'],
                G['mmask'][ix_t], G['gt'][ix_t], G['label'][ix_t],
                G['resid'][ix_t])

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Head(K, T).to(dev)
    anchors = anchors.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    def evaluate(ix):
        model.eval()
        with torch.no_grad():
            L, R, ACC = [], [], []
            for s in range(0, len(ix), 256):
                b = batch(ix[s:s + 256])
                ego, ag, am, mp, mm, gt, lab, res = [x.to(dev) for x in b]
                logits, resid = model(ego, ag, am, mp, mm, anchors)
                k = logits.argmax(-1)
                pred = anchors[k] + resid
                L.append(((pred - gt) ** 2).sum(-1).mean(1).cpu())
                R.append((((pred[:, -1] - gt[:, -1]) ** 2).sum(-1) ** 0.5).cpu())
                ACC.append((k == lab).float().mean().cpu())
        model.train()
        return (torch.cat(L).numpy(), torch.cat(R).numpy(),
                float(torch.stack(ACC).mean()))

    best_fde, best_ep, patience = 1e9, 0, 0
    best_state = None
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        perm = rng.permutation(tr)
        tot = 0.0
        tt_data = tt_fwd = tt_bwd = 0.0
        for s in range(0, len(perm), args.bs):
            ix = perm[s:s + args.bs]
            tt = time.time()
            ego, ag, am, mp, mm, gt, lab, res = batch(ix)
            tt_data += time.time() - tt
            tt = time.time()
            logits, resid = model(ego, ag, am, mp, mm, anchors, k_hint=lab)
            tt_fwd += time.time() - tt
            tt = time.time()
            loss = F.cross_entropy(logits, lab, weight=cw.to(dev))
            loss = loss + 5.0 * (resid - res).abs().mean()
            pred = anchors[lab] + resid
            loss = loss + 2.0 * (pred - gt).abs().mean()
            loss.backward(); opt.step(); opt.zero_grad()
            tt_bwd += time.time() - tt
            tot += float(loss) * len(ix)
        sched.step()
        if ep % 5 == 0 or ep == 1:
            mse, fde, acc = evaluate(va)
            ade = np.sqrt(mse)
            cur = float(np.median(fde))
            tag = ''
            if cur < best_fde - 1e-4:
                best_fde, best_ep = cur, ep
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
                patience = 0
                tag = '  *best*'
            else:
                patience += 1
            print(f"ep {ep:4d} loss {tot/len(tr):.3f} | val ADE {ade.mean():.2f} "
                  f"FDE {fde.mean():.2f} p50 {cur:.2f} "
                  f"({time.time()-t0:.0f}s) acc {acc*100:.1f}%{tag}", flush=True)
            if patience >= args.patience:
                print(f'early stop at ep {ep} (best ep {best_ep})', flush=True)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    mse, fde, acc = evaluate(va)
    ade = np.sqrt(mse)
    print(f"BEST(ep {best_ep}) ADE {ade.mean():.3f}  FDE {fde.mean():.3f}  "
          f"p50 {np.median(fde):.3f}  <1m {100*np.mean(fde<1):.0f}%  "
          f"cls_acc {acc*100:.1f}%")
    if args.out:
        torch.save({'model': model.state_dict(), 'anchors': anchors.cpu(),
                    'K': K, 'T': T}, args.out)
        print('saved', args.out)


if __name__ == '__main__':
    main()
