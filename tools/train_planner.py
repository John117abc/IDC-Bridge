"""上层规划器训练：WTA + 组内优势加权回归（IAWR）+ 多样性。

v1 说明
- 训练内奖励用**快速代理**（碰撞/出界/进度，全部 numpy 向量化），
  验证时用 PDMS（tools/planner_reward.py）对齐；
- 纯追踪 rollout 先省略（v1 直接用候选点序列近似执行），
  §6.3 会测"代理 vs 实际执行器"的差异。
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '/gpudrive/baselines/IDC-Bridge/src')

EGO_L, EGO_W = 4.9, 1.9
AG_L, AG_W = 4.5, 1.9
K_DEF, G_DEF = 6, 4


# ---------------- 模型 ----------------
class Planner(nn.Module):
    def __init__(self, K=K_DEF, d=256, layers=4, heads=8, dropout=0.1):
        super().__init__()
        self.K = K
        self.ego_proj = nn.Linear(23, d)
        self.ag_proj = nn.Linear(28, d)
        self.mp_proj = nn.Linear(7, d)
        self.type_emb = nn.Parameter(torch.zeros(3, d))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=4 * d,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, layers)
        self.queries = nn.Parameter(torch.randn(K, d) * 0.02)
        self.cross = nn.MultiheadAttention(d, heads, dropout=dropout,
                                           batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 16)

    def forward(self, ego, ag, amask, mp, mmask):
        B = ego.shape[0]
        tok = torch.cat([self.ego_proj(ego).unsqueeze(1),
                         self.ag_proj(ag), self.mp_proj(mp)], 1)
        tok = tok + self.type_emb[None, :1, :].expand(
            -1, 1, -1).repeat(1, 1, 1)
        types = torch.cat([
            torch.zeros(1, dtype=torch.long), torch.ones(ag.shape[1], dtype=torch.long),
            torch.full((mp.shape[1],), 2, dtype=torch.long)]).to(ego.device)
        tok = tok + self.type_emb[types][None]
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=ego.device),
                         amask < 0.5, mmask < 0.5], 1)
        F = self.encoder(tok, src_key_padding_mask=pad)
        q = self.queries[None].expand(B, -1, -1)
        out, _ = self.cross(q, F, F, key_padding_mask=pad)
        out = self.norm(q + out)
        out = out + self.ffn(out)
        return self.head(out).view(B, self.K, 8, 2)


# ---------------- 快速奖励（numpy） ----------------
def densify(anchor, n=40):
    t_src = np.linspace(0, 4.0, len(anchor))
    t_dst = np.linspace(0, 4.0, n)
    return np.stack([np.interp(t_dst, t_src, anchor[:, 0]),
                     np.interp(t_dst, t_src, anchor[:, 1])], 1)


def ego_to_world(pts, p0, h0):
    c, s = np.cos(h0), np.sin(h0)
    R = np.array([[c, -s], [s, c]], np.float64)
    return pts.astype(np.float64) @ R.T + p0[None]


def seg_dist(P, A, B):
    """P (T,2) 到线段集 (S,2) 的最小距离，返回 (T,)。"""
    if len(A) == 0:
        return np.full(len(P), 1e9)
    AB = B - A
    den = np.maximum((AB ** 2).sum(1), 1e-12)
    out = np.empty(len(P))
    for i, p in enumerate(P):
        ap = p[None] - A
        t = np.clip((ap * AB).sum(1) / den, 0, 1)
        Q = A + t[:, None] * AB
        out[i] = np.min(np.linalg.norm(Q - p[None], axis=1))
    return out


def obb_hit(p, hp, q, hq):
    """两矩形是否重叠（分离轴，简化版：投影到两轴）。"""
    def corners(c, h, L, W):
        c_, s_ = np.cos(h), np.sin(h)
        dx = np.array([[L / 2, W / 2], [L / 2, -W / 2],
                       [-L / 2, -W / 2], [-L / 2, W / 2]])
        R = np.array([[c_, -s_], [s_, c_]])
        return dx @ R.T + c
    A = corners(p, hp, EGO_L, EGO_W)
    B = corners(q, hq, AG_L, AG_W)
    for ax in (np.array([np.cos(hp), np.sin(hp)]),
               np.array([-np.sin(hp), np.cos(hp)]),
               np.array([np.cos(hq), np.sin(hq)]),
               np.array([-np.sin(hq), np.cos(hq)])):
        a = A @ ax
        b = B @ ax
        if a.max() < b.min() or b.max() < a.min():
            return False
    return True


def head_of(traj):
    d = np.gradient(traj, axis=0)
    return np.arctan2(d[:, 1], d[:, 0])


def fast_reward(traj_w, agf, agv, A_e, B_e, expert_w):
    """返回 (reward, coll, off, prog)。"""
    T = min(len(traj_w), agf.shape[0])
    hd = head_of(traj_w[:T])
    coll = 0
    for t in range(T):
        d = np.linalg.norm(agf[t] - traj_w[t][None], axis=1)
        d = np.where(agv[t] > 0, d, 1e9)
        if d.min() > 7.0:
            continue
        for n in range(agf.shape[1]):
            if agv[t, n] <= 0:
                continue
            if np.linalg.norm(agf[t, n] - traj_w[t]) > 7.0:
                continue
            hq = hd[t] if t + 1 >= T else np.arctan2(
                *(agf[min(t + 1, T - 1), n] - agf[t, n])[::-1])
            if obb_hit(traj_w[t], hd[t], agf[t, n], hq):
                coll = 1
                break
        if coll:
            break
    dd = seg_dist(traj_w[:T], A_e, B_e)
    off = 1 if dd.min() < 1.075 else 0
    # 进度：终点在专家路径上的单调投影
    seg = np.diff(expert_w, axis=0)
    sl = np.linalg.norm(seg, axis=1) + 1e-9
    cum = np.r_[0.0, np.cumsum(sl)]
    idx, best = 0, 0.0
    for p in traj_w[:T]:
        i1 = min(len(expert_w) - 1, idx + 25)
        if i1 <= idx:
            break
        A_, B_ = expert_w[idx:i1], expert_w[idx + 1:i1 + 1]
        AB = B_ - A_
        ap = p[None] - A_
        tt = np.clip((ap * AB).sum(1) / np.maximum((AB ** 2).sum(1), 1e-12), 0, 1)
        Q = A_ + tt[:, None] * AB
        k = int(np.argmin(np.linalg.norm(Q - p[None], axis=1)))
        s = cum[idx + k] + tt[k] * sl[idx + k]
        best = max(best, s)
        idx = idx + k
    prog = float(min(1.2, best / (cum[-1] + 1e-9)))
    r = (1 - coll) * (1 - off) * (0.5 + 0.5 * prog)
    return r, coll, off, prog


# ---------------- 训练 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/workspace/data/planner5k')
    ap.add_argument('--out', default='/workspace/data/planner_ckpt')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--G', type=int, default=G_DEF)
    ap.add_argument('--sigma', type=float, default=0.5)
    ap.add_argument('--lam_iawr', type=float, default=0.1)
    ap.add_argument('--lam_div', type=float, default=0.01)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--check2', action='store_true')
    args = ap.parse_args()
    torch.manual_seed(0)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    z = np.load(f'{args.data}/train.npz', allow_pickle=True)
    N = len(z['ego'])
    if args.limit and args.limit < N:
        N = args.limit
    rng = np.random.default_rng(0)
    ego = torch.tensor(z['ego'], dtype=torch.float32)
    ag = torch.tensor(z['agent'], dtype=torch.float32)
    amask = torch.tensor(z['amask'], dtype=torch.float32)
    mp = torch.tensor(z['map'], dtype=torch.float32)
    mmask = torch.tensor(z['mmask'], dtype=torch.float32)
    gt = torch.tensor(z['gt'], dtype=torch.float32)
    K = args.G and K_DEF
    model = Planner(K=K).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    print(f'样本 {N}  K={K}  G={args.G}  device={dev}')

    if args.check2:
        check2(model, opt, z, dev, args)
        return

    for ep in range(1, args.epochs + 1):
        perm = np.random.permutation(N)
        tot_wta = tot_iawr = tot_div = 0.0
        n_iawr = 0
        t0 = time.time()
        for b0 in range(0, N, args.bs):
            idx = perm[b0:b0 + args.bs]
            ego_b = ego[idx].to(dev)
            ag_b = ag[idx].to(dev)
            am_b = amask[idx].to(dev)
            mp_b = mp[idx].to(dev)
            mm_b = mmask[idx].to(dev)
            gt_b = gt[idx].to(dev)
            pred = model(ego_b, ag_b, am_b, mp_b, mm_b)          # (B,K,8,2)
            d = (pred - gt_b[:, None]).abs().mean(-1).mean(-1)   # (B,K)
            wta = d.min(1).values.mean()
            w = (1 / (d + 1e-6))
            w = w / w.sum(1, keepdim=True)
            div = (w[:, :, None] * w[:, None, :] *
                   (pred[:, :, None, :, :] - pred[:, None, :, :, :]).abs().mean(-1).mean(-1)
                   ).sum(1).sum(1).mean() / (K * (K - 1) / 2)
            loss = wta + args.lam_div * (-div)
            # IAWR（v1：用 gt 派生候选 + 模型 top-1 的 jitter，奖励离线算）
            iawr, n_upd = (None, 0)
            if args.lam_iawr > 0:
                iawr, n_upd = iawr_loss(pred, idx, z, args, dev, rng)
            if iawr is not None:
                loss = loss + args.lam_iawr * iawr
                tot_iawr += float(iawr)
                n_iawr += n_upd
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot_wta += float(wta)
        nb = int(np.ceil(N / args.bs))
        print(f'ep {ep:3d} wta={tot_wta/nb:.4f} iawr={tot_iawr/max(nb,1):.4f} '
              f'upd_groups={n_iawr} ({time.time()-t0:.0f}s)')
        os.makedirs(args.out, exist_ok=True)
        torch.save({'model': model.state_dict(), 'K': K}, f'{args.out}/latest.pt')


def iawr_loss(pred, idx, z, args, dev, rng):
    """按 (样本, 模态) 配对：L = Σ w_{k,g}·||pred_k − sg(τ̃_{k,g})||₁。"""
    B, K = pred.shape[0], pred.shape[1]
    ramp = (np.linspace(0, 1, 40) ** 2)[:, None]
    terms, wsum, n_groups = [], 0.0, 0
    for j in range(B):
        i = int(idx[j])
        ego_w = z['ego_world'][i]
        agf, agv = z['agent_fut'][i], z['agent_fut_valid'][i]
        A_e, B_e = road_edges(z['map'][i], z['mmask'][i])
        expert_w = ego_to_world(densify(z['gt'][i]), ego_w[:2], ego_w[2])
        for k in range(K):
            base = densify(pred[j, k].detach().cpu().numpy())
            rs, tg = [], []
            for g in range(args.G):
                eps = rng.normal(0, 1, (2,))
                cj = base + args.sigma * ramp * eps[None]
                tw = ego_to_world(cj, ego_w[:2], ego_w[2])
                r, _, _, _ = fast_reward(tw, agf, agv, A_e, B_e, expert_w)
                rs.append(r)
                tg.append(cj)
            rs = np.array(rs, dtype=np.float64)
            if rs.std() < 1e-6:
                continue
            adv = (rs - rs.mean()) / (rs.std() + 1e-6)
            w = np.maximum(adv, 0.0)
            if w.sum() <= 1e-6:
                continue
            n_groups += 1
            for g in range(args.G):
                if w[g] <= 0:
                    continue
                ww = float(w[g] / w.sum())
                # 40 点（0.1s）里取 8 个 waypoint（0.5s）
                tgt8 = torch.tensor(tg[g][[0, 5, 10, 15, 20, 25, 30, 35]],
                                        dtype=torch.float32, device=dev)
                terms.append(ww * (pred[j, k] - tgt8).abs().mean())
                wsum += ww
    if not terms:
        return None, 0
    return torch.stack(terms).sum() / max(wsum, 1e-6), n_groups


def road_edges(mp, mmask):
    segs = []
    for i in range(len(mp)):
        if mmask[i] <= 0 or mp[i, 5] > 0.5:
            continue
        x, y, ln, c, s = mp[i, 0], mp[i, 1], mp[i, 2], mp[i, 3], mp[i, 4]
        segs.append((np.array([x - 0.5 * ln * c, y - 0.5 * ln * s]),
                     np.array([x + 0.5 * ln * c, y + 0.5 * ln * s])))
    if not segs:
        return np.zeros((0, 2)), np.zeros((0, 2))
    return (np.stack([a for a, _ in segs]), np.stack([b for _, b in segs]))


def check2(model, opt, z, dev, args):
    """检查点 2：固定一条预测，给某变体正权重，一步更新后应向其靠近。"""
    i = 0
    ego_w = z['ego_world'][i]
    ramp = (np.linspace(0, 1, 40) ** 2)[:, None]
    x = torch.tensor(z['ego'][i:i + 1], dtype=torch.float32, device=dev)
    pred = model(x, torch.tensor(z['agent'][i:i + 1], dtype=torch.float32, device=dev),
                 torch.tensor(z['amask'][i:i + 1], dtype=torch.float32, device=dev),
                 torch.tensor(z['map'][i:i + 1], dtype=torch.float32, device=dev),
                 torch.tensor(z['mmask'][i:i + 1], dtype=torch.float32, device=dev))
    base = pred.reshape(-1, 8, 2).detach()
    # 目标：在 8 点空间给一个固定横向偏移（模拟"更好的变体"）
    off = torch.tensor([0.0, float(args.sigma)], dtype=torch.float32, device=dev)
    target = (base + off[None, None]).detach()
    for step in range(20):
        pred = model(x, torch.tensor(z['agent'][i:i + 1], dtype=torch.float32, device=dev),
                     torch.tensor(z['amask'][i:i + 1], dtype=torch.float32, device=dev),
                     torch.tensor(z['map'][i:i + 1], dtype=torch.float32, device=dev),
                     torch.tensor(z['mmask'][i:i + 1], dtype=torch.float32, device=dev))
        d0 = (pred.reshape(-1, 8, 2) - base).abs().mean()
        d1 = (pred.reshape(-1, 8, 2) - target).abs().mean()
        loss = d1.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step in (0, 19):
            print(f'step {step}: 到 base {float(d0):.4f}  到 target {float(d1):.4f}')
    print('检查点 2：若"到 target"持续下降、"到 base"上升 → 更新方向正确')


if __name__ == '__main__':
    main()
