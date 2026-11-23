"""IDC 策略封装（CARLA 版）：加载 checkpoint → 30 帧状态窗口 → actor → 物理动作。

口径（全部抄自 src/agents/idc_agent.py）:
  ENCODED_DIM = 4 + 32 + 8 + 18 + 1 + 1 = 64
  _encode(state_mode='ego')：输入已是自车系（自车放原点、航向 0，周车相对）
  RunningNorm：用 input_norm.sum/sumsq 反推 mean/std（checkpoint 里带统计）
  动作: delta = clip(u0*0.3, ±0.6) rad;  acc = tanh(u1)*{1.5 | -3.0}
"""
import math
import os
from collections import deque

import numpy as np
import torch
import torch.nn as nn

CKPT_DEFAULT = '/mnt/2T_HDD/WaymoDatabas/checkpoints/20261119/idc-waymo-v1.0_examples_041839_episode=475.pth'
STEER_SCALE, STEER_MAX = 0.3, 0.6
ACC_MAX_POS, ACC_MAX_NEG = 1.5, 3.0
ENCODED_DIM = 64
WINDOW = 30
DT = 0.1
WHEELBASE = 5.0


class RunningNorm(nn.Module):
    """与训练一致的输入归一化（mean/std 由 sum/sumsq/count 推出）。"""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.register_buffer('sum', torch.zeros(dim))
        self.register_buffer('sumsq', torch.zeros(dim))
        self.register_buffer('count', torch.zeros(1))

    def load_from_state_dict_stats(self, sum_, sumsq, count):
        self.sum.copy_(torch.as_tensor(sum_, dtype=torch.float32))
        self.sumsq.copy_(torch.as_tensor(sumsq, dtype=torch.float32))
        self.count.copy_(torch.as_tensor([count], dtype=torch.float32))

    @property
    def mean(self):
        c = torch.clamp(self.count, min=1.0)
        return self.sum / c

    @property
    def std(self):
        c = torch.clamp(self.count, min=1.0)
        var = torch.clamp(self.sumsq / c - self.mean ** 2, min=0.0)
        return torch.sqrt(var + self.eps)

    def forward(self, x):
        return (x - self.mean) / self.std


def to_ego_state(state_world, ego_pose):
    """世界系 66 维 → 自车系（自车放原点、航向 0，周车转相对）。"""
    x, y, h = float(ego_pose[0]), float(ego_pose[1]), float(ego_pose[2])
    s = np.array(state_world, np.float32, copy=True)
    c, sn = math.cos(-h), math.sin(-h)
    s[0], s[1] = 0.0, 0.0
    s[4] = 0.0
    o = s[6:6 + 32].reshape(8, 4)
    for k in range(8):
        dx, dy = float(o[k, 0]) - x, float(o[k, 1]) - y
        rx = dx * c - dy * sn
        ry = dx * sn + dy * c
        o[k, 0], o[k, 1] = rx, ry
        o[k, 2] = math.atan2(math.sin(float(o[k, 2]) - h),
                             math.cos(float(o[k, 2]) - h))
    s[6:6 + 32] = o.reshape(-1)
    return s


def encode(s64):
    """等价于 idc_agent._encode(state_mode='ego')：ego 6→4，丢 v_lat/omega。"""
    assert s64.shape[-1] == 66, s64.shape
    ego4 = s64[..., [0, 1, 2, 4]]
    others = s64[..., 6:6 + 32]
    rest = s64[..., 6 + 32:]
    return np.concatenate([ego4, others, rest], -1)


class IDCPolicy:
    def __init__(self, ckpt=CKPT_DEFAULT, device='cuda:0'):
        self.dev = torch.device(device if torch.cuda.is_available() else 'cpu')
        ck = torch.load(ckpt, map_location='cpu', weights_only=False)
        sd = ck['model_state_dict']
        # actor 结构：transformer（config 里 actor_type='transformer'）
        cfg = ck.get('config')
        d_model = int(getattr(cfg, 'transformer_d_model', 256))
        nhead = int(getattr(cfg, 'transformer_nhead', 4))
        layers = int(getattr(cfg, 'transformer_num_layers', 2))
        drop = float(getattr(cfg, 'transformer_dropout', 0.1))
        self.actor = _TransformerActor(ENCODED_DIM, d_model, nhead, layers, drop)
        self.actor.load_state_dict(sd['actor'] if 'actor' in sd else sd,
                                   strict=False)
        self.actor.to(self.dev).eval()

        self.norm = RunningNorm(ENCODED_DIM)
        inn = ck.get('input_norm', {})
        if inn:
            cnt = float(inn.get('count', inn.get('n', 1.0)))
            self.norm.load_from_state_dict_stats(inn['sum'], inn['sumsq'],
                                                 cnt if cnt > 0 else 1.0)
        self.norm.to(self.dev)
        self.win = deque(maxlen=WINDOW)
        self.state_dim = int(ck.get('state_dim', 66))
        self.state_mode = ck.get('state_mode', 'ego')
        self.ref_mode = ck.get('ref_mode', 'nearest')
        print(f'[IDC] ckpt 载入: state_dim={self.state_dim} mode={self.state_mode} '
              f'ref={self.ref_mode} actor=transformer d={d_model}', flush=True)

    def reset(self):
        self.win.clear()

    @torch.no_grad()
    def act(self, state_world, ego_pose, deterministic=True):
        """输入世界系 66 维状态 + 自车位姿，返回 (steer_norm, acc, delta_rad)。"""
        s = to_ego_state(state_world, ego_pose)
        e = encode(s).astype(np.float32)
        self.win.append(e)
        w = np.tile(self.win[0], (WINDOW, 1)).astype(np.float32)
        w[-len(self.win):] = np.array(self.win)
        x = torch.from_numpy(w)[None].to(self.dev)
        x = self.norm(x)
        u = self.actor(x)[0].cpu().numpy()          # [2] = [steer_raw, acc_raw]
        delta = float(np.clip(u[0] * STEER_SCALE, -STEER_MAX, STEER_MAX))
        acc_n = float(np.tanh(u[1]))
        acc = acc_n * (ACC_MAX_POS if acc_n >= 0 else ACC_MAX_NEG)
        steer_norm = float(np.clip(delta / STEER_MAX, -1.0, 1.0))
        return steer_norm, acc, delta


class _TransformerActor(nn.Module):
    """与 IDC Actor 的 transformer 版本同构：Linear→Transformer→Linear(2)."""

    def __init__(self, state_dim, d_model, nhead, layers, dropout):
        super().__init__()
        self.inp = nn.Linear(state_dim, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                         dim_feedforward=4 * d_model,
                                         dropout=dropout, batch_first=True,
                                         norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.out = nn.Linear(d_model, 2)

    def forward(self, x):
        h = self.inp(x)
        h = self.enc(h)
        return self.out(h[:, -1])
