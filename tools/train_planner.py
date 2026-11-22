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

from metrics.openloop_pdms_vec import geo_from_npz, score_batch  # noqa
from metrics.openloop_pdms_torch import (geo_from_npz_t,  # noqa
                                         score_batch_t)

_INTERP = {}


def _interp_mat(dev):
    """densify 用的线性插值矩阵 (40,8)，与 densify() 完全等价。"""
    key = str(dev)
    if key not in _INTERP:
        t_src = np.linspace(0, 4.0, 8)
        t_dst = np.linspace(0, 4.0, 40)
        M = np.zeros((40, 8), np.float32)
        for k, t in enumerate(t_dst):
            j = int(np.searchsorted(t_src, t, side='right') - 1)
            j = min(max(j, 0), 6)
            w = (t - t_src[j]) / (t_src[j + 1] - t_src[j])
            M[k, j] = 1 - w
            M[k, j + 1] = w
        _INTERP[key] = torch.tensor(M, dtype=torch.float64, device=dev)
    return _INTERP[key]

EGO_L, EGO_W = 4.9, 1.9
AG_L, AG_W = 4.5, 1.9
K_DEF, G_DEF = 6, 4
# 每条模态的横向偏置（m，终点处），乘 --iawr_bias 使用；顺序 = 模态序号
BIAS_LAT = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])


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
        self.cls = nn.Linear(d, 1)      # 模态置信度头（WTA 赢家当标签）
        self.emit_logits = False
        # 锚点模态（方案 B）：全零表示不用（向后兼容，eval 脚本无需改动）
        self.register_buffer('anchors', torch.zeros(K, 8, 2))
        self.res_scale = 3.0

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
        pred = self.head(out).view(B, self.K, 8, 2)
        if float(self.anchors.abs().sum()) > 0:
            # 锚点 + 有界残差：模态之间不可能重合
            pred = self.anchors[None] + self.res_scale * torch.tanh(pred)
        logits = self.cls(out).squeeze(-1)
        if self.emit_logits:
            return pred, logits
        return pred


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
    ap.add_argument('--div', default='weighted', choices=['weighted', 'hinge'],
                    help='多样性项：weighted=旧版加权平均距离；hinge=终点/中点 margin 推远')
    ap.add_argument('--div_margin', type=float, default=2.0,
                    help='hinge 多样性的终点 margin（m）')
    ap.add_argument('--iawr_bias', type=float, default=0.0,
                    help='IAWR 的模态专属横向偏置缩放（0=关闭；1.0=±2 m 档）')
    ap.add_argument('--qinit', default='zero', choices=['zero', 'anchor'],
                    help='query 初始化：zero=原随机小值；anchor=k-means 锚点投影')
    ap.add_argument('--anchor-modes', action='store_true',
                    help='方案 B：pred = anchor_k + 3·tanh(残差)，锚点由训练集 GT 无监督聚类')
    ap.add_argument('--res-scale', type=float, default=3.0,
                    help='锚点残差幅度上限（m）')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--reward', default='fast',
                    choices=['fast', 'pdms', 'pdms_gpu'])
    ap.add_argument('--workers', type=int, default=0,
                    help='奖励并行 worker 数（0=单进程；pdms 建议 8）')
    ap.add_argument('--allow-cpu', action='store_true',
                    help='允许在 CPU 上训练（默认禁止，避免 GPU 掉了却静默退回 CPU）')
    ap.add_argument('--init-from', default='',
                    help='从已有 checkpoint 继续训练（容器重启/中断后用）')
    ap.add_argument('--check2', action='store_true')
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit('[FATAL] CUDA 不可用（容器 GPU 掉了？）。'
                         '如确实要用 CPU 训练，加 --allow-cpu')
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    reward_fn = reward_pdms if args.reward == 'pdms' else reward_fast
    global _Z, _POOL

    # 一次性解压到内存：压缩 npz 是惰性解压的，fork 后多进程并发读会踩坏 zlib 流
    z = dict(np.load(f'{args.data}/train.npz', allow_pickle=True))
    _Z = z
    if args.workers > 1 and args.reward == 'pdms':
        import multiprocessing as mp
        _POOL = mp.get_context('fork').Pool(args.workers)
    N = len(z['ego'])
    if args.limit and args.limit < N:
        N = args.limit
    rng = np.random.default_rng(args.seed)
    ego = torch.tensor(z['ego'], dtype=torch.float32)
    ag = torch.tensor(z['agent'], dtype=torch.float32)
    amask = torch.tensor(z['amask'], dtype=torch.float32)
    mp = torch.tensor(z['map'], dtype=torch.float32)
    mmask = torch.tensor(z['mmask'], dtype=torch.float32)
    gt = torch.tensor(z['gt'], dtype=torch.float32)
    K = args.G and K_DEF
    model = Planner(K=K).to(dev)
    if args.anchor_modes:
        from build_head_dataset import kmeans
        gts = np.ascontiguousarray(z['gt'].reshape(len(z['gt']), -1), np.float32)
        C, _ = kmeans(gts, K * 3, seed=0)            # 多聚一些，再挑互相分散的 K 条
        A = C.reshape(-1, 8, 2)
        sel = [0]
        while len(sel) < K:
            ep = A[:, -1]
            d = np.min(np.linalg.norm(ep[:, None] - ep[sel][None], axis=2), axis=1)
            sel.append(int(np.argmax(d)))
        A = A[sel]
        with torch.no_grad():
            model.anchors.copy_(torch.tensor(A, device=dev))
        model.res_scale = float(args.res_scale)
        ep = A[:, -1]
        dd = [np.linalg.norm(ep[i] - ep[j]) for i in range(K) for j in range(i + 1, K)]
        print(f'[anchor] K={K} 终点两两距离 min={min(dd):.1f} 中位={np.median(dd):.1f} m '
              f'残差界±{args.res_scale} m')
    if args.qinit == 'anchor':
        # 无监督：对训练集 GT 未来做 k-means，把锚点投影成 query 初值（之后仍可学习）
        from build_head_dataset import kmeans
        Xg = np.ascontiguousarray(z['gt'].reshape(len(z['gt']), -1), np.float32)
        C, _ = kmeans(Xg, K, seed=0)
        A0 = torch.tensor(C.reshape(K, 16), dtype=torch.float32)
        A0 = A0 / (A0.abs().max() + 1e-6)
        proj = torch.randn(16, model.queries.shape[1]) / 4.0
        q0 = A0 @ proj
        q0 = q0 / (q0.norm(dim=1, keepdim=True) + 1e-6) * 0.5
        with torch.no_grad():
            model.queries.copy_(q0.to(dev))
        print('[qinit] 用 k-means 锚点初始化 query')
    if args.init_from:
        ck0 = torch.load(args.init_from, map_location='cpu')
        model.load_state_dict(ck0['model'], strict=False)
        print(f'[resume] 载入 {args.init_from}')
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    print(f'样本 {N}  K={K}  G={args.G}  reward={args.reward}  device={dev}')

    if args.check2:
        check2(model, opt, z, dev, args)
        return

    for ep in range(1, args.epochs + 1):
        perm = np.random.permutation(N)
        tot_wta = tot_iawr = tot_div = 0.0
        n_iawr = 0
        agg = {'r': 0.0, 'std': 0.0, 'n': 0, 'std0': 0, 'pos': 0, 'gtot': 0,
               'rmin': 1e9, 'rmax': -1e9}
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
            if args.div == 'weighted':
                # 旧版：按"离 GT 越近权重越大"加权，最大化两两平均 L1
                w = (1 / (d + 1e-6))
                w = w / w.sum(1, keepdim=True)
                div = (w[:, :, None] * w[:, None, :] *
                       (pred[:, :, None, :, :] - pred[:, None, :, :, :]).abs().mean(-1).mean(-1)
                       ).sum(1).sum(1).mean() / (K * (K - 1) / 2)
                loss = wta + args.lam_div * (-div)
            else:
                # hinge：终点/中点两两距离小于 margin 才推，超过就不管（不鼓励乱跑）
                iu = torch.triu_indices(K, K, offset=1)
                ep_t = pred[:, :, -1]                            # (B,K,2)
                md_t = pred[:, :, 4]                             # (B,K,2) ~2 s
                pair_ep = torch.cdist(ep_t, ep_t)[:, iu[0], iu[1]]
                pair_md = torch.cdist(md_t, md_t)[:, iu[0], iu[1]]
                h = (torch.clamp(args.div_margin - pair_ep, min=0.0).mean()
                     + 0.5 * torch.clamp(0.6 * args.div_margin - pair_md,
                                         min=0.0).mean())
                loss = wta + args.lam_div * h
            # IAWR：对每个模态做 G 个时间平滑扰动，按组内优势加权回归
            iawr, n_upd, st = (None, 0, None)
            if args.lam_iawr > 0:
                if args.reward == 'pdms_gpu':
                    iawr, n_upd, st = _iawr_loss_gpu(pred, idx, z, args, dev, rng)
                else:
                    iawr, n_upd, st = iawr_loss(pred, idx, z, args, dev, rng,
                                                reward_fn)
                if st and st['n'] > 0:
                    for kk in ('r', 'std', 'n', 'std0', 'pos', 'gtot'):
                        agg[kk] += st[kk]
                    agg['rmin'] = min(agg['rmin'], st['rmin'])
                    agg['rmax'] = max(agg['rmax'], st['rmax'])
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
        msg = f'ep {ep:3d} wta={tot_wta/nb:.4f}'
        if args.lam_iawr > 0:
            n = max(agg['n'], 1)
            ng = max(agg['std0'] + n_iawr, 1)
            msg += (f' iawr={tot_iawr/max(nb,1):.4f} upd={n_iawr}'
                    f' R={agg["r"]/n:.1f}[{agg["rmin"]:.0f},{agg["rmax"]:.0f}]'
                    f' Rstd={agg["std"]/n:.1f}'
                    f' std0={100*agg["std0"]/ng:.0f}%'
                    f' pos={100*agg["pos"]/max(agg["gtot"],1):.0f}%')
        msg += f' ({time.time()-t0:.0f}s)'
        print(msg, flush=True)
        os.makedirs(args.out, exist_ok=True)
        torch.save({'model': model.state_dict(), 'K': K}, f'{args.out}/latest.pt')


def iawr_loss(pred, idx, z, args, dev, rng, reward_fn):
    """按 (样本, 模态) 配对：L = Σ w_{k,g}·||pred_k − sg(τ̃_{k,g})||₁。

    reward_fn(z, i, cands) -> (K*G,) 奖励；cands 是自车系 (K*G,40,2)。
    奖励计算可并行（fork 出的 worker 共享父进程的 z 内存）。
    """
    B, K = pred.shape[0], pred.shape[1]
    ramp = (np.linspace(0, 1, 40) ** 2)[:, None]
    terms, wsum, n_groups = [], 0.0, 0
    st = {'r': 0.0, 'std': 0.0, 'n': 0, 'std0': 0, 'pos': 0, 'gtot': 0,
          'rmin': 1e9, 'rmax': -1e9}

    # ---- 1. 生成候选 ----
    tasks, all_cands = [], []
    for j in range(B):
        i = int(idx[j])
        base = pred[j].detach().cpu().numpy()                   # (K,8,2) 自车系
        cands = np.empty((K * args.G, 40, 2))
        for k in range(K):
            b40 = densify(base[k])
            eps = rng.normal(0, 1.0, (args.G, 2))
            # 模态专属横向偏置：让每条模态探索不同方向，抵抗 IAWR 把模态拉到一起
            bk = b40
            if args.iawr_bias > 0:
                d = np.gradient(b40, axis=0)
                n = np.stack([-d[:, 1], d[:, 0]], 1)
                n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
                lat = BIAS_LAT[k % len(BIAS_LAT)] * args.iawr_bias
                bk = b40 + lat * ramp * n
            cands[k * args.G:(k + 1) * args.G] = (
                bk[None] + args.sigma * ramp[None] * eps[:, None, :])
        tasks.append((i, cands))
        all_cands.append(cands)

    # ---- 2. 算奖励（可并行）----
    if _POOL is not None and reward_fn is reward_pdms:
        rss = _POOL.map(_worker_reward, tasks)
    else:
        rss = [reward_fn(z, i, c) for (i, c) in tasks]

    # ---- 3. 组内优势 + 加权回归 ----
    for j in range(B):
        rs = np.asarray(rss[j], np.float64)
        if rs.size != K * args.G or not np.all(np.isfinite(rs)):
            continue
        rs = rs.reshape(K, args.G)
        cands = all_cands[j]
        tgt8 = cands[:, [0, 5, 10, 15, 20, 25, 30, 35]]          # (K*G,8,2)
        st['r'] += float(rs.mean()); st['n'] += 1
        st['rmin'] = min(st['rmin'], float(rs.min()))
        st['rmax'] = max(st['rmax'], float(rs.max()))
        for k in range(K):
            r = rs[k]
            st['std'] += float(r.std())
            if r.std() < 1e-6:
                st['std0'] += 1
                continue
            adv = (r - r.mean()) / (r.std() + 1e-6)
            w = np.maximum(adv, 0.0)
            if w.sum() <= 1e-6:
                continue
            n_groups += 1
            st['pos'] += int((w > 0).sum())
            st['gtot'] += args.G
            for g in range(args.G):
                if w[g] <= 0:
                    continue
                ww = float(w[g] / w.sum())
                # 40 点（0.1s）里取 8 个 waypoint（0.5s）
                t8 = torch.tensor(tgt8[k * args.G + g], dtype=torch.float32,
                                  device=dev)
                terms.append(ww * (pred[j, k] - t8).abs().mean())
                wsum += ww
    if not terms:
        return None, 0, st
    return torch.stack(terms).sum() / max(wsum, 1e-6), n_groups, st


def _iawr_loss_gpu(pred, idx, z, args, dev, rng):
    """IAWR 的 GPU 版：候选生成 + PDMS 打分全部在 GPU 上。"""
    B, K = pred.shape[0], pred.shape[1]
    ramp = torch.tensor((np.linspace(0, 1, 40) ** 2)[:, None],
                        dtype=torch.float64, device=dev)
    It = _interp_mat(dev)
    terms, wsum, n_groups = [], 0.0, 0
    st = {'r': 0.0, 'std': 0.0, 'n': 0, 'std0': 0, 'pos': 0, 'gtot': 0,
          'rmin': 1e9, 'rmax': -1e9}
    for j in range(B):
        i = int(idx[j])
        base = pred[j].detach().double()                     # (K,8,2)
        p40 = torch.einsum('tw,kwd->ktd', It, base)          # (K,40,2)
        eps = torch.randn(K, args.G, 2, dtype=torch.float64, device=dev)
        cands = p40[:, None] + args.sigma * eps[:, :, None, :] * ramp[None, None]
        if args.iawr_bias > 0:
            d = torch.gradient(p40, dim=1)[0]                # (K,40,2)
            nrm = torch.stack([-d[..., 1], d[..., 0]], -1)
            nrm = nrm / (nrm.norm(dim=-1, keepdim=True) + 1e-9)
            lat = (torch.tensor(BIAS_LAT[:K], dtype=torch.float64, device=dev)
                   .view(K, 1, 1) * args.iawr_bias)          # (K,1,1)
            bias = lat * ramp[None]                          # (K,40,1)
            cands = cands + bias[:, None] * nrm[:, None]
        geo = geo_from_npz_t(z, i, dev)
        rs = score_batch_t(geo, cands.reshape(K * args.G, 40, 2))['driving_score']
        rs = rs.detach().cpu().numpy().reshape(K, args.G)
        if not np.all(np.isfinite(rs)):
            continue
        tgt8 = cands.reshape(K * args.G, 40, 2)[:, [0, 5, 10, 15, 20, 25, 30, 35]]
        st['r'] += float(rs.mean()); st['n'] += 1
        st['rmin'] = min(st['rmin'], float(rs.min()))
        st['rmax'] = max(st['rmax'], float(rs.max()))
        for k in range(K):
            r = rs[k]
            st['std'] += float(r.std())
            if r.std() < 1e-6:
                st['std0'] += 1
                continue
            adv = (r - r.mean()) / (r.std() + 1e-6)
            w = np.maximum(adv, 0.0)
            if w.sum() <= 1e-6:
                continue
            n_groups += 1
            st['pos'] += int((w > 0).sum())
            st['gtot'] += args.G
            for g in range(args.G):
                if w[g] <= 0:
                    continue
                ww = float(w[g] / w.sum())
                t8 = tgt8[k * args.G + g].to(torch.float32)
                terms.append(ww * (pred[j, k] - t8).abs().mean())
                wsum += ww
    if not terms:
        return None, 0, st
    return torch.stack(terms).sum() / max(wsum, 1e-6), n_groups, st


def reward_pdms(z, i, cands):
    """向量化 PDMS（与评测同一套公式；几何来自数据集的紧凑表示）。"""
    geo = geo_from_npz(z, i)
    return score_batch(geo, cands)['driving_score']


_Z = None        # 数据集（fork 后 worker 共享，不用重复加载）
_POOL = None     # 奖励并行池


def _worker_reward(arg):
    i, cands = arg
    return reward_pdms(_Z, i, cands)


def reward_fast(z, i, cands):
    """v1 快速代理（碰撞/出界/进度），保留作对照。

    注意：agent_fut 存的是 (N,T)，fast_reward 要 (T,N)，这里显式转置
    （旧版没转，属于 bug，A/B 对照时需知道）。
    """
    ego_w = z['ego_world'][i]
    agf = np.asarray(z['agent_fut'][i], np.float64).transpose(1, 0, 2)
    agv = np.asarray(z['agent_fut_valid'][i], np.float64).transpose(1, 0)
    A_e, B_e = road_edges(z['map'][i], z['mmask'][i])
    expert_w = ego_to_world(densify(z['gt'][i]), ego_w[:2], ego_w[2])
    out = np.empty(len(cands))
    for m, c in enumerate(cands):
        tw = ego_to_world(c, ego_w[:2], ego_w[2])
        r, _, _, _ = fast_reward(tw, agf, agv, A_e, B_e, expert_w)
        out[m] = r
    return out


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
