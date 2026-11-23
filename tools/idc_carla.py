"""CARLA 侧的 IDC 推理封装（84 维状态 → steer/throttle/brake）

设计要点
--------
1. 状态构造复用 `idc_state_carla.build_state`（66 维 v1 核心）+ `frames.neighbor_extras`
   （追加 18 维）= 84 维，与 GPUDrive 训练时的布局逐位一致；
2. 网络部分**直接复用仓库里的类**（`TransformerActor`、`RunningNorm`），不重写；
3. 归一化保持训练口径：前 64 维过 RunningNorm(64)，后 18 维透传；
4. 动作映射做轴距换算：IDC 训练用 L=5.0 m 的自行车模型，CARLA 车轴距约 2.9 m，
   δ_carla = atan(L_carla · tan(δ_idc) / L_idc)，再按 CARLA 最大转向角归一化。
"""
import math
import os
import sys
from collections import deque

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_SRC_CANDIDATES = [os.path.join(_REPO, 'src'), _HERE,
                   os.environ.get('VEC_IDC_SRC', ''),
                   '/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/src']
for _p in _SRC_CANDIDATES:
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

import importlib.util as _ilu


def _load_module(name, relpath):
    """按文件路径加载单个模块（避免经包 __init__ 连带导入 gpudrive 依赖）。"""
    for base in _SRC_CANDIDATES:
        if not base:
            continue
        p = os.path.join(base, relpath)
        if os.path.isfile(p):
            spec = _ilu.spec_from_file_location(name, p)
            mod = _ilu.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise ImportError(f'找不到 {relpath}')


_cac = _load_module('_idc_cac', os.path.join('models', 'continuous_actor_critic.py'))
_rn = _load_module('_idc_running_norm', os.path.join('utils', 'running_norm.py'))
TransformerActor = _cac.TransformerActor
RunningNorm = _rn.RunningNorm
import frames as F                                            # noqa: E402
import idc_state_carla as S                                   # noqa: E402

L_IDC = 5.0          # IDC 训练时的轴距
L_CARLA = 2.9        # CARLA 默认车辆轴距（可按实际车调）
WINDOW = 30          # 状态窗口（10 Hz）


class IDCCarlaPolicy:
    def __init__(self, ckpt, device='cuda:0', max_steer_deg=70.0):
        self.dev = torch.device(device if torch.cuda.is_available() else 'cpu')
        ck = torch.load(ckpt, map_location='cpu', weights_only=False)
        sd = ck['model_state_dict']
        actor_sd = sd['actor'] if isinstance(sd, dict) and 'actor' in sd else sd

        cfg = ck.get('config')
        d_model = int(getattr(cfg, 'transformer_d_model', 256) or 256)
        nhead = int(getattr(cfg, 'transformer_nhead', 4) or 4)
        layers = int(getattr(cfg, 'transformer_num_layers', 2) or 2)
        drop = float(getattr(cfg, 'transformer_dropout', 0.1) or 0.1)
        hidden = int(getattr(cfg, 'actor_hidden', 256) or 256)

        win = int(getattr(cfg, 'window_size', WINDOW) or WINDOW)
        self.action_space = str(ck.get('action_space', 'delta_acc'))
        self.off_max = float(ck.get('off_max', 3.0))
        print(f'[idc] action_space={self.action_space} off_max={self.off_max}',
              flush=True)
        self.actor = TransformerActor(
            state_dim=82, d_model=d_model, nhead=nhead,
            num_layers=layers, window_size=win, dropout=drop).to(self.dev).eval()
        print(f'[idc] actor: d_model={d_model} nhead={nhead} layers={layers} '
              f'window={win}', flush=True)

        # 容忍式加载（首层 64→82 时做列拷贝热启动）
        own = self.actor.state_dict()
        ok, expand = {}, []
        for k, v in actor_sd.items():
            if k in own and own[k].shape == v.shape:
                ok[k] = v
            elif (k in own and v.dim() == 2 and own[k].dim() == 2
                  and v.shape[0] == own[k].shape[0] and v.shape[1] < own[k].shape[1]):
                new = torch.zeros_like(own[k])
                new[:, :v.shape[1]] = v
                ok[k] = new
                expand.append((k, tuple(v.shape), tuple(own[k].shape)))
            else:
                print(f'[idc] 跳过参数 {k}', flush=True)
        self.actor.load_state_dict(ok, strict=False)
        if expand:
            print(f'[idc] 热启动扩展 {expand}', flush=True)

        self.norm = RunningNorm(64)
        inn = ck.get('input_norm')
        if inn:
            self.norm.load_state_dict(inn)
            print(f'[idc] 载入归一化统计 n={self.norm.count:.0f}', flush=True)

        self.max_steer = math.radians(max_steer_deg)
        self.win = deque(maxlen=WINDOW)
        self.state_dim = int(ck.get('state_dim', 84))
        print(f'[idc] ckpt={os.path.basename(ckpt)} state_dim={self.state_dim} '
              f'd_model={d_model} layers={layers}', flush=True)

    def reset(self):
        self.win.clear()

    @torch.no_grad()
    def act(self, state84):
        e = F.encode82(state84).astype(np.float32)
        self.win.append(e)
        w = np.tile(self.win[0], (WINDOW, 1)).astype(np.float32)
        w[-len(self.win):] = np.array(self.win)
        x = torch.from_numpy(w)[None].to(self.dev)
        base = self.norm(x[..., :64])
        x = torch.cat([base, x[..., 64:]], dim=-1)
        u = self.actor(x)[0].cpu().numpy()      # [steer_raw, acc_raw]
        delta = float(np.clip(u[0] * 0.3, -0.6, 0.6))
        acc_n = float(np.tanh(u[1]))
        acc = acc_n * (1.5 if acc_n >= 0 else 3.0)
        return delta, acc


def to_carla_control(delta_idc, acc, max_steer_deg=70.0,
                     acc_pos_max=1.5, acc_neg_max=3.0, steer_gain=1.0):
    """IDC 的 (前轮角, 加速度) → CARLA (steer, throttle, brake)。

    steer_gain: 补偿 CARLA 转向响应延迟/限幅导致的“转向不足”（稳态横向误差）
    """
    delta_carla = math.atan(L_CARLA * math.tan(delta_idc) / L_IDC)
    steer = float(np.clip(delta_carla * steer_gain / math.radians(max_steer_deg),
                          -1.0, 1.0))
    if acc >= 0:
        thr = float(np.clip(acc / acc_pos_max, 0.0, 1.0))
        brk = 0.0
    else:
        thr = 0.0
        brk = float(np.clip(-acc / acc_neg_max, 0.0, 1.0))
    return steer, thr, brk


def make_state84(ego_pose_canon, speed, neighbors_canon, path, ref_idx=0,
                 veh_len=4.9, a_lat=0.0, steer_prev=0.0):
    """CARLA 场景 → 84 维状态（输入必须是 canonical 坐标）。"""
    _nbr = [[float(q['x']), float(q['y']), float(q['h']), float(q['v'])]
            for q in neighbors_canon]
    core66, t = S.build_state(ego_pose_canon, speed, _nbr, path,
                              ref_idx=ref_idx, veh_len=veh_len)
    extras = np.concatenate([
        np.array([a_lat / F.SCALE_A_LAT, float(steer_prev)], np.float32),
        F.neighbor_extras(ego_pose_canon, neighbors_canon, v_ego=speed),
    ]).astype(np.float32)
    st = np.concatenate([core66[:65], extras, core66[65:]]).astype(np.float32)
    assert st.shape[0] == 84, st.shape
    return st, t
