"""在 CARLA 采集的 (84 维状态, 纯追踪动作) 上，用行为克隆微调 IDC 的 actor。

特点：
  · 状态窗口 30 帧、padding 方式与部署完全一致（tile 首帧）
  · 监督目标是**物理量**（前轮角 δ_idc、加速度 a），不是 raw u，避免 atanh 边界问题
  · 其余网络参数从原 checkpoint 热启动（首层 64→82 列拷贝），只训 actor
  · 输出 checkpoint 格式与 IDCCarlaPolicy 兼容

用法：
  python tools/bc_finetune.py --data /mnt/2T_HDD/plant2_vector/bc \
      --ckpt /mnt/2T_HDD/WaymoDatabas/checkpoints/20261123/xxx.pth \
      --out /mnt/2T_HDD/WaymoDatabas/checkpoints/bc/idc_bc.pth --epochs 20
"""
import argparse
import glob
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_REPO = '/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge'
for _p in (os.path.join(_REPO, 'src'), os.path.join(_REPO, 'tools')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.continuous_actor_critic import TransformerActor  # noqa: E402
import frames as FR                                            # noqa: E402

L_IDC, L_CARLA = 5.0, 2.9
WINDOW = 30
STEER_MAX = 0.6
STEER_SCALE = 0.3
ACC_POS, ACC_NEG = 1.5, 3.0


def build_episodes(data_dir):
    eps = []
    for f in sorted(glob.glob(os.path.join(data_dir, 'bc_*.npz'))):
        z = np.load(f)
        S, A = z['state'], z['action']
        if len(S) < 10:
            continue
        eps.append((S.astype(np.float32), A.astype(np.float32)))
        print(f'  {os.path.basename(f)}: {S.shape[0]} 帧')
    return eps


def actions_to_targets(A, max_steer_deg):
    """CARLA (steer, thr, brk) → (δ_idc, acc)。

    注意符号：部署时 steer_carla = -δ_carla/max_steer（STEER_SIGN=-1），
    所以反推必须把负号补回来，否则学到的转向是反的。
    """
    max_steer = math.radians(max_steer_deg)
    delta_carla = -A[:, 0] * max_steer
    delta_idc = np.arctan(L_IDC * np.tan(delta_carla) / L_CARLA)
    acc = A[:, 1] * ACC_POS - A[:, 2] * ACC_NEG
    return np.stack([delta_idc, acc], 1).astype(np.float32)


def make_windows(S, idx, window=WINDOW):
    """按部署口径构造窗口：不足补首帧（tile）。"""
    out = np.zeros((len(idx), window, S.shape[1]), np.float32)
    for j, i in enumerate(idx):
        lo = max(0, i - window + 1)
        seg = S[lo:i + 1]
        w = np.tile(seg[0], (window, 1))
        w[-len(seg):] = seg
        out[j] = w
    return out


def encode82_torch(s):
    """84 → 82：去绝对 x,y，周车转自车系，保留追加特征（与部署 _encode 一致）。"""
    lead = s.shape[:-1]
    x, y, phi = s[..., 0], s[..., 1], s[..., 4]
    z = torch.zeros_like(x)
    ego4 = torch.stack([z, z, s[..., 2], z], dim=-1)
    o = s[..., 6:38].reshape(*lead, 8, 4)
    c, sn = torch.cos(phi), torch.sin(phi)
    dx = o[..., 0] - x.unsqueeze(-1)
    dy = o[..., 1] - y.unsqueeze(-1)
    rx = dx * c.unsqueeze(-1) + dy * sn.unsqueeze(-1)
    ry = -dx * sn.unsqueeze(-1) + dy * c.unsqueeze(-1)
    dphi = torch.atan2(torch.sin(o[..., 2] - phi.unsqueeze(-1)),
                       torch.cos(o[..., 2] - phi.unsqueeze(-1)))
    others = torch.stack([rx, ry, dphi, o[..., 3]], dim=-1).reshape(
        *lead, 32)
    rest = s[..., 38:65]
    veil = s[..., 83:84]
    base64 = torch.cat([ego4, others, rest, veil], dim=-1)
    extras = s[..., 65:83]
    return torch.cat([base64, extras], dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/mnt/2T_HDD/plant2_vector/bc')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--max_steer_deg', type=float, default=70.0)
    args = ap.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device =', dev)

    eps = build_episodes(args.data)
    if not eps:
        raise SystemExit('没有 BC 数据')

    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    cfg = ck.get('config')
    d_model = int(getattr(cfg, 'transformer_d_model', 256) or 256)
    nhead = int(getattr(cfg, 'transformer_nhead', 4) or 4)
    layers = int(getattr(cfg, 'transformer_num_layers', 2) or 2)
    drop = float(getattr(cfg, 'transformer_dropout', 0.1) or 0.1)
    win = int(getattr(cfg, 'window_size', WINDOW) or WINDOW)

    actor = TransformerActor(state_dim=82, d_model=d_model, nhead=nhead,
                             num_layers=layers, window_size=win,
                             dropout=drop).to(dev)
    base_sd = ck['model_state_dict']['actor'] if 'actor' in ck['model_state_dict'] \
        else ck['model_state_dict']
    own = actor.state_dict()
    ok, expand = {}, []
    for k, v in base_sd.items():
        if k in own and own[k].shape == v.shape:
            ok[k] = v
        elif (k in own and v.dim() == 2 and own[k].dim() == 2
              and v.shape[0] == own[k].shape[0] and v.shape[1] < own[k].shape[1]):
            nv = torch.zeros_like(own[k]); nv[:, :v.shape[1]] = v
            ok[k] = nv; expand.append(k)
    actor.load_state_dict(ok, strict=False)
    print('热启动扩展:', expand)

    opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    # 预先把窗口与目标缓存
    cache = []
    for S, A in eps:
        T = actions_to_targets(A, args.max_steer_deg)
        idx = np.arange(0, len(S), 1)
        cache.append((make_windows(S, idx, win), T))
    X = np.concatenate([c[0] for c in cache])
    Y = np.concatenate([c[1] for c in cache])
    print('样本数', X.shape, Y.shape)
    print('目标统计: delta 中位 %.3f (p10 %.3f p90 %.3f)  acc 中位 %.2f (p10 %.2f p90 %.2f)'
          % (np.median(Y[:, 0]), np.percentile(Y[:, 0], 10), np.percentile(Y[:, 0], 90),
             np.median(Y[:, 1]), np.percentile(Y[:, 1], 10), np.percentile(Y[:, 1], 90)))
    print('目标方差: delta %.4f  acc %.4f' % (np.var(Y[:, 0]), np.var(Y[:, 1])))

    # 归一化：用 **BC 数据自身**的统计（旧 checkpoint 统计与 CARLA 分布不符，
    # 会把输入压到饱和区，模型只能学到均值）。部署时从 checkpoint 读同一份统计。
    allb = torch.cat([encode82_torch(torch.from_numpy(X[i:i + 256]))[..., :64]
                      for i in range(0, len(X), 256)]).numpy()
    mean = allb.mean(0)
    std = np.maximum(allb.std(0), 1e-3)
    _n = float(len(allb))
    print('BC 归一化统计: mean[:3]=%s std[:3]=%s'
          % (np.round(mean[:3], 3), np.round(std[:3], 3)))
    mean_t = torch.tensor(mean, dtype=torch.float32, device=dev)
    std_t = torch.tensor(std, dtype=torch.float32, device=dev)
    _new_norm = {'sum': torch.tensor(mean, dtype=torch.float64),
                 'sumsq': torch.tensor((std ** 2 + mean ** 2) * _n,
                                       dtype=torch.float64),
                 'n': _n, 'dim': 64}

    Xt = torch.from_numpy(X)
    Yt = torch.from_numpy(Y)
    N = len(Xt)
    for ep in range(args.epochs):
        perm = torch.randperm(N)
        tot, nb = 0.0, 0
        for i in range(0, N, args.batch):
            idx = perm[i:i + args.batch]
            xb = Xt[idx].to(dev)
            yb = Yt[idx].to(dev)
            xb = encode82_torch(xb)
            x = torch.cat([(xb[..., :64] - mean_t) / std_t, xb[..., 64:]], -1)
            u = actor(x)
            delta = torch.clamp(u[:, 0] * STEER_SCALE, -STEER_MAX, STEER_MAX)
            an = torch.tanh(u[:, 1])
            acc = torch.where(an >= 0, an * ACC_POS, an * ACC_NEG)
            pred = torch.stack([delta, acc], 1)
            loss = F.mse_loss(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        print(f'epoch {ep+1}/{args.epochs}  loss={tot/max(nb,1):.5f}', flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_ck = dict(ck)
    msd = dict(ck['model_state_dict'])
    msd['actor'] = actor.state_dict()
    out_ck['model_state_dict'] = msd
    out_ck['state_dim'] = 84
    out_ck['input_norm'] = _new_norm          # 保存 BC 统计，部署时自动使用
    torch.save(out_ck, args.out)
    print('saved', args.out)


if __name__ == '__main__':
    main()
