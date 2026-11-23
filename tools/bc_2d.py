"""步骤 3：把 IDC 的动作空间蒸馏成 (横向偏移, 加速度意图)。

输入：CARLA BC 数据（state 84 维, action=[steer, thr, brk]）
标签反推：
    δ_carla = -steer · max_steer                （STEER_SIGN=-1）
    off     = clip(-(L_idc/L_carla)·d_prev·tan(δ_carla), ±3)   # 米，左正
    acc     = thr·1.5 - brk·3.0
输出：checkpoint（含 action_space='off_acc' 标记），可直接被 IDCCarlaPolicy 加载。
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

L_IDC, L_CARLA = 5.0, 2.9
WINDOW = 30
OFF_MAX = 3.0
ACC_POS, ACC_NEG = 1.5, 3.0
D_PREV = 5.0


def encode82_torch(s):
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
    others = torch.stack([rx, ry, dphi, o[..., 3]], dim=-1).reshape(*lead, 32)
    rest = s[..., 38:65]
    veil = s[..., 83:84]
    base = torch.cat([ego4, others, rest, veil], dim=-1)
    return torch.cat([base, s[..., 65:83]], dim=-1)


def make_windows(S, window=WINDOW):
    out = np.zeros((len(S), window, S.shape[1]), np.float32)
    for i in range(len(S)):
        lo = max(0, i - window + 1)
        seg = S[lo:i + 1]
        w = np.tile(seg[0], (window, 1))
        w[-len(seg):] = seg
        out[i] = w
    return out


def labels_from_actions(A, max_steer_deg):
    max_steer = math.radians(max_steer_deg)
    d_carla = -A[:, 0] * max_steer
    off = -(L_IDC / L_CARLA) * D_PREV * np.tan(d_carla)
    off = np.clip(off, -OFF_MAX, OFF_MAX)
    acc = A[:, 1] * ACC_POS - A[:, 2] * ACC_NEG
    return np.stack([off, acc], 1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/mnt/2T_HDD/plant2_vector/bc')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=8e-4)
    ap.add_argument('--max_steer_deg', type=float, default=70.0)
    args = ap.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device =', dev)

    Xs, Ys = [], []
    for f in sorted(glob.glob(os.path.join(args.data, 'bc_*.npz'))):
        z = np.load(f)
        S, A = z['state'].astype(np.float32), z['action'].astype(np.float32)
        if len(S) < 10:
            continue
        Xs.append(make_windows(S))
        Ys.append(labels_from_actions(A, args.max_steer_deg))
        print(f'  {os.path.basename(f)}: {len(S)} 帧')
    X = np.concatenate(Xs)
    Y = np.concatenate(Ys)
    print('样本', X.shape, Y.shape)
    print('标签: off 中位 %.2f (p10 %.2f p90 %.2f)  acc 中位 %.2f (p10 %.2f p90 %.2f)'
          % (np.median(Y[:, 0]), np.percentile(Y[:, 0], 10), np.percentile(Y[:, 0], 90),
             np.median(Y[:, 1]), np.percentile(Y[:, 1], 10), np.percentile(Y[:, 1], 90)))

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
    base = ck['model_state_dict']['actor']
    own = actor.state_dict()
    ok = {k: v for k, v in base.items() if k in own and own[k].shape == v.shape}
    actor.load_state_dict(ok, strict=False)
    print('热启动参数量', len(ok), '/', len(own))

    # 归一化统计沿用 CARLA BC 数据
    allb = torch.cat([encode82_torch(torch.from_numpy(X[i:i + 256]))[..., :64]
                      for i in range(0, len(X), 256)]).numpy()
    mean, std = allb.mean(0), np.maximum(allb.std(0), 1e-3)
    mean_t = torch.tensor(mean, dtype=torch.float32, device=dev)
    std_t = torch.tensor(std, dtype=torch.float32, device=dev)
    _n = float(len(allb))

    Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y)
    # 标签归一化到 ~O(1)
    Yn = Yt.clone()
    Yn[:, 0] = Yn[:, 0] / OFF_MAX
    Yn[:, 1] = Yn[:, 1] / 3.0
    opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    N = len(Xt)
    for ep in range(args.epochs):
        perm = torch.randperm(N)
        tot, nb = 0.0, 0
        for i in range(0, N, args.batch):
            idx = perm[i:i + args.batch]
            xb = encode82_torch(Xt[idx].to(dev))
            x = torch.cat([(xb[..., :64] - mean_t) / std_t, xb[..., 64:]], -1)
            u = actor(x)
            pred = torch.stack([torch.clamp(u[:, 0] / OFF_MAX, -1, 1),
                                torch.clamp(u[:, 1] / 3.0, -1, 1)], 1)
            loss = F.mse_loss(pred, Yn[idx].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f'epoch {ep+1}/{args.epochs} loss={tot/max(nb,1):.5f}', flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out = dict(ck)
    msd = dict(ck['model_state_dict']); msd['actor'] = actor.state_dict()
    out['model_state_dict'] = msd
    out['state_dim'] = 84
    out['action_space'] = 'off_acc'          # 部署端据此切换
    out['off_max'] = OFF_MAX
    out['input_norm'] = {'sum': torch.tensor(mean * _n, dtype=torch.float64),
                         'sumsq': torch.tensor((std ** 2 + mean ** 2) * _n,
                                               dtype=torch.float64),
                         'n': _n, 'dim': 64}
    torch.save(out, args.out)
    print('saved', args.out, '(action_space=off_acc)')


if __name__ == '__main__':
    main()
