"""GPU 版 PDMS 打分（torch），与 openloop_pdms_vec.score_batch 逐项对齐。

用途：IAWR 训练时把 24 条候选的六项指标全部放在 GPU 上算，
避免 numpy 单样本循环把训练卡成 CPU-bound。

对齐口径：与 numpy 版（已对 score_trajectory 验证到 4e-14）比较，
要求 max |ΔPDMS| < 1e-4。
"""
import numpy as np
import torch

EPS = 1e-6
OFFROAD_MARGIN = 1.0


# ---------------- 几何 ----------------
def _rot_to_ego_t(pts, p0, h0, device):
    d = pts - torch.tensor(p0, dtype=torch.float64, device=device)
    c, s = np.cos(-h0), np.sin(-h0)
    return torch.stack([d[..., 0] * c - d[..., 1] * s,
                        d[..., 0] * s + d[..., 1] * c], -1)


def geo_from_npz_t(z, i, device='cuda'):
    """数据集第 i 条 → 自车系几何（全部 torch.float64 张量）。"""
    x, y, h0 = (float(v) for v in z['ego_world'][i])
    n = int(np.sum(z['amask'][i] > 0.5))
    dev = torch.device(device)
    af = torch.as_tensor(z['agent_fut'][i, :n], dtype=torch.float64, device=dev)
    ap = _rot_to_ego_t(af, (x, y), h0, dev)                       # (A,T,2)
    av = torch.as_tensor(z['agent_fut_valid'][i, :n] > 0.5, device=dev)
    ah = torch.as_tensor(z['agent_fut_head'][i, :n], dtype=torch.float64,
                         device=dev) - h0
    ad = torch.stack([torch.as_tensor(z['agent'][i, :n, 5], dtype=torch.float64,
                                      device=dev),
                      torch.as_tensor(z['agent'][i, :n, 6], dtype=torch.float64,
                                      device=dev)], 1)

    m = int(z['n_map_r'][i]) if 'n_map_r' in z else 0
    seg = torch.as_tensor(z['map_r'][i, :m], dtype=torch.float64, device=dev)
    is_lane = seg[:, 4] > 0.5
    ea, eb = seg[~is_lane][:, :2], seg[~is_lane][:, 2:4]
    la, lb = seg[is_lane][:, :2], seg[is_lane][:, 2:4]

    ef = torch.as_tensor(z['expert40'][i], dtype=torch.float64, device=dev)
    expert_fut = _rot_to_ego_t(ef, (x, y), h0, dev)
    expert_head = torch.as_tensor(z['expert_head40'][i], dtype=torch.float64,
                                  device=dev) - h0
    hm = {k: torch.as_tensor(z['hm_' + k][i], dtype=torch.float64, device=dev)
          for k in ('ttc', 'jerk', 'lat')}
    for k in ('coll', 'off', 'ddc'):
        hm[k] = torch.as_tensor(z['hm_' + k][i] > 0.5, device=dev)
    ego_dim = torch.as_tensor(z['ego_dim'][i], dtype=torch.float64, device=dev)
    return dict(ap=ap, av=av, ah=ah, ad=ad, ea=ea, eb=eb, la=la, lb=lb,
                expert_fut=expert_fut, expert_head=expert_head, hm=hm,
                ego_dim=ego_dim, n_future=len(hm['coll']))


# ---------------- 基础算子 ----------------
def seg_min_dist_t(P, A, B):
    """P (M,2) 到线段集 (K,2) 的最小距离 → (M,)。"""
    if A.numel() == 0:
        return torch.full((P.shape[0],), 1e9, dtype=P.dtype, device=P.device)
    AB = B - A                                     # (K,2)
    den = (AB ** 2).sum(1).clamp_min(1e-12)        # (K,)
    diff = P[:, None, :] - A[None]                  # (M,K,2)
    t = ((diff * AB[None]).sum(2) / den[None]).clamp(0.0, 1.0)
    proj = A[None] + t[..., None] * AB[None]
    return (P[:, None, :] - proj).pow(2).sum(2).clamp_min(0).sqrt().min(1)[0]


def _corners(p, h, dim):
    """p (M,T,A,2), h (M,T,A), dim ()/(2,)/(A,2) → 角点 (M,T,A,4,2)。"""
    Lb, Wb = dim[..., 0] / 2, dim[..., 1] / 2
    if Lb.dim() == 1:                     # (A,) → 便于广播
        Lb, Wb = Lb[None, None, :], Wb[None, None, :]
    lx = torch.tensor([-1., 1., 1., -1.], dtype=p.dtype, device=p.device)
    ly = torch.tensor([-1., -1., 1., 1.], dtype=p.dtype, device=p.device)
    ox = lx * Lb[..., None]               # (...,4)
    oy = ly * Wb[..., None]
    c, s = torch.cos(h), torch.sin(h)
    x = ox * c[..., None] - oy * s[..., None]
    y = ox * s[..., None] + oy * c[..., None]
    return torch.stack([x + p[..., 0, None], y + p[..., 1, None]], -1)


def _obb_hit(pe, he, de, pa, ha, da):
    """向量化 SAT：pe (M,T,A,2) he (M,T,A) pa (M,T,A,2) ha (M,T,A) → (M,T,A) bool。"""
    ce = _corners(pe, he, de)                       # (M,T,A,4,2)
    ca = _corners(pa, ha, da)
    axes = []
    for hh in (he, ha):
        c, s = torch.cos(hh), torch.sin(hh)
        axes.append(torch.stack([c, s], -1))
        axes.append(torch.stack([-s, c], -1))
    hit = torch.ones(pe.shape[:-1], dtype=torch.bool, device=pe.device)
    for ax in axes:                                  # 4 个轴（每箱 2 个）
        pe_p = (ce * ax[..., None, :]).sum(-1)        # (M,T,A,4)
        pa_p = (ca * ax[..., None, :]).sum(-1)
        sep = (pe_p.max(-1)[0] < pa_p.min(-1)[0]) | \
              (pa_p.max(-1)[0] < pe_p.min(-1)[0])
        hit = hit & ~sep
    return hit


def _heading_speed(traj, dt=0.1):
    M, T = traj.shape[:2]
    d = traj[:, 1:] - traj[:, :-1] if T > 1 else torch.zeros(
        (M, 1, 2), dtype=traj.dtype, device=traj.device)
    speed = torch.zeros((M, T), dtype=traj.dtype, device=traj.device)
    speed[:, :T - 1] = d.norm(dim=-1) / dt
    heading = torch.zeros((M, T), dtype=traj.dtype, device=traj.device)
    heading[:, :T - 1] = torch.atan2(d[..., 1], d[..., 0])
    if T > 1:
        heading[:, -1] = heading[:, -2]
    return heading, speed


def _ratio(bad, good):
    """good 为 0/inf 处记 1（与 numpy 版一致）。bad (M,T) good (T,)。"""
    g = good[None]
    r = (g / bad.clamp_min(EPS)).clamp(0.0, 1.0)
    valid = torch.isfinite(g) & (g > EPS)
    return torch.where(valid, r, torch.ones_like(r))


def _ttc_score(cand, human):
    c_fin = torch.isfinite(cand)
    h_fin = torch.isfinite(human)
    both = c_fin & h_fin
    out = torch.ones_like(cand)
    out = torch.where(both, (cand / human.clamp_min(EPS)).clamp(0, 1), out)
    out = torch.where(h_fin & ~c_fin, torch.ones_like(out), out)
    out = torch.where(~h_fin & c_fin, torch.zeros_like(out), out)
    return out


def _progress(g, q):
    A_, B_ = g['expert_fut'][:-1], g['expert_fut'][1:]
    if A_.shape[0] == 0:
        return torch.zeros(q.shape[0], dtype=q.dtype, device=q.device)
    AB = B_ - A_
    den = (AB ** 2).sum(1).clamp_min(1e-12)
    L = AB.norm(dim=1)
    diff = q[:, None, :] - A_[None]
    t = ((diff * AB[None]).sum(2) / den[None]).clamp(0.0, 1.0)
    proj = A_[None] + t[..., None] * AB[None]
    j = (q[:, None, :] - proj).pow(2).sum(2).argmin(1)
    cum = torch.cumsum(L, 0)
    arc = cum[j] - L[j] + t[torch.arange(q.shape[0], device=q.device), j] * L[j]
    return arc / max(float(L.sum()), 1e-6)


# ---------------- 主入口 ----------------
def score_batch_t(g, trajs):
    """trajs (M,T,2) 自车系 → dict of (M,) 张量。"""
    if trajs.dim() == 2:
        trajs = trajs[None]
    M, T = trajs.shape[:2]
    T = int(min(T, g['n_future']))
    trajs = trajs[:, :T]
    hm = g['hm']
    heading, speed = _heading_speed(trajs)

    A = g['ap'].shape[0]
    if A > 0:
        ap_f = g['ap'].transpose(0, 1)                    # (T,A,2)
        av_f = g['av'].transpose(0, 1)
        ah_f = g['ah'].transpose(0, 1)
        dist = (trajs[:, :, None, :] - ap_f[None]).norm(dim=-1)   # (M,T,A)
        coll = _obb_hit(trajs[:, :, None, :].expand(M, T, A, 2),
                        heading[:, :, None].expand(M, T, A),
                        g['ego_dim'], ap_f[None].expand(M, T, A, 2),
                        ah_f[None].expand(M, T, A), g['ad'])
        coll = coll & (dist < 6.0) & av_f[None]
        coll = coll.any(2)

        dv = ap_f[None] - trajs[:, :, None, :]                     # (M,T,A,2)
        d2 = dv.norm(dim=-1)
        fwd = dv[..., 0] * torch.cos(heading)[:, :, None] + \
            dv[..., 1] * torch.sin(heading)[:, :, None]
        ok = (av_f[None] & (d2 > 1e-3) & (d2 <= 60.0)
              & (fwd >= 0.5 * d2) & (speed[:, :, None] > 1e-6))
        ttc = torch.where(ok, d2 / speed[:, :, None].clamp_min(1e-12),
                          torch.full_like(d2, float('inf'))).min(2)[0]
    else:
        coll = torch.zeros((M, T), dtype=torch.bool, device=trajs.device)
        ttc = torch.full((M, T), float('inf'), dtype=trajs.dtype,
                         device=trajs.device)

    P = trajs.reshape(-1, 2)
    off = (seg_min_dist_t(P, g['ea'], g['eb']) < OFFROAD_MARGIN).reshape(M, T)
    lat = (seg_min_dist_t(P, g['la'], g['lb']).reshape(M, T) if g['la'].numel() > 0
           else torch.zeros((M, T), dtype=trajs.dtype, device=trajs.device))
    jerk = torch.zeros((M, T), dtype=trajs.dtype, device=trajs.device)
    if T > 1:
        jerk[:, 1:] = (speed[:, 1:] - speed[:, :-1]).abs() / 0.1

    dh = heading - g['expert_head'][:T][None]
    ddc = torch.abs(torch.atan2(torch.sin(dh), torch.cos(dh))) > np.pi / 2

    hm_coll, hm_off, hm_ddc = hm['coll'][:T], hm['off'][:T], hm['ddc'][:T]
    nc = torch.where((coll & ~hm_coll[None]).any(1), 0.0, 1.0)
    dac = torch.where((off & ~hm_off[None]).any(1), 0.0, 1.0)
    ddc_g = torch.where((ddc & ~hm_ddc[None]).any(1), 0.0, 1.0)

    ttc_s = _ttc_score(ttc, hm['ttc'][:T][None].expand(M, T))
    comfort = _ratio(jerk, hm['jerk'][:T])
    comfort = torch.where((hm['jerk'][:T][None] <= EPS) & (jerk <= EPS),
                          1.0, comfort)
    comfort = torch.where((hm['jerk'][:T][None] <= EPS) & (jerk > EPS),
                          0.0, comfort)
    lk = _ratio(lat, hm['lat'][:T])
    lk = torch.where((hm['lat'][:T][None] <= EPS) & (lat <= EPS), 1.0, lk)
    lk = torch.where((hm['lat'][:T][None] <= EPS) & (lat > EPS), 0.0, lk)

    ep = _progress(g, trajs[:, -1]).clamp(max=1.0)
    ttc_avg, c_avg, lk_avg = ttc_s.mean(1), comfort.mean(1), lk.mean(1)
    weighted = (5 * ep + 5 * ttc_avg + 2 * c_avg + 2 * lk_avg) / 14.0
    score = nc * dac * ddc_g * weighted * 100.0
    return dict(driving_score=score, nc=nc, dac=dac, ddc=ddc_g, ep=ep,
                ttc=ttc_avg, comfort=c_avg, lk=lk_avg, coll=coll, off=off)
