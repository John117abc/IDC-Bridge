"""IDC-Bridge 修改自检：不依赖 GPUDrive / Waymo 数据，用假 env + 假 builder
把 idc_agent 的改动路径全部跑一遍（缩放 / 掩码 / 截断 / IS 权重 / 约束归属）。

用法（在 IDC-Bridge 根目录）：
    python tools/selftest_idc.py
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))


def _stub_gpudrive_if_missing():
    """自检不需要 GPUDrive（用假 env），但 utils/__init__ 会 import 可视化模块
    间接依赖 gpudrive。缺失时注入最小桩，保持对真实包的零修改。"""
    try:
        import gpudrive  # noqa: F401
        return
    except Exception:
        pass
    import types
    g = types.ModuleType('gpudrive'); g.__path__ = []
    v = types.ModuleType('gpudrive.visualize'); v.__path__ = []
    u = types.ModuleType('gpudrive.visualize.utils'); u.__path__ = []
    u.img_from_fig = lambda fig: None
    g.visualize = v
    v.utils = u
    # src/env/__init__ -> env_utils -> gpudrive.env.config.EnvConfig
    e = types.ModuleType('gpudrive.env'); e.__path__ = []
    ec = types.ModuleType('gpudrive.env.config')

    class _EnvConfig:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    ec.EnvConfig = _EnvConfig
    g.env = e
    e.config = ec
    sys.modules.update({'gpudrive': g, 'gpudrive.visualize': v,
                        'gpudrive.visualize.utils': u,
                        'gpudrive.env': e, 'gpudrive.env.config': ec})


_stub_gpudrive_if_missing()

from agents.idc_agent import DiscreteIDCAgent  # noqa: E402
from buffer import PERBuffer  # noqa: E402,F401
from env.idc_state_builder import GPUDriveObservationBuilder  # noqa: E402
from utils.running_norm import RunningNorm  # noqa: E402

DIM_EGO, DIM_OTH, DIM_VAL, DIM_REF, DIM_T, DIM_VEH = 6, 32, 8, 18, 1, 1
TOTAL = DIM_EGO + DIM_OTH + DIM_VAL + DIM_REF + DIM_T + DIM_VEH
REF0 = DIM_EGO + DIM_OTH + DIM_VAL          # 46
TEMP0 = REF0 + DIM_REF                      # 64
VEH0 = TEMP0 + 1                            # 65


class FakeBuilder:
    EXPERT_TRAJ_LEN = 91

    def __init__(self, W):
        t = torch.arange(91).float()
        self.ref_tensor = torch.zeros(W, 1, 1, 91, 8)
        self.ref_tensor[:, 0, 0, :, 0] = t * 1.0     # pos_x（每索引 1 m）
        self.ref_tensor[:, 0, 0, :, 1] = 0.0         # pos_y
        self.ref_tensor[:, 0, 0, :, 2] = 10.0        # speed
        self.ref_tensor[:, 0, 0, :, 3] = 0.0         # heading
        self.ref_tensor[:, 0, 0, :, 4] = 5.0         # road_dist
        self.ref_tensor[:, 0, 0, :, 5] = 0.0         # curv_ahead
        self.ref_tensor[:, 0, 0, :, 6] = 10.0        # spd_ahead
        self.ref_tensor[:, 0, 0, :, 7] = 5.0         # road_ahead
        self.ref_mode = 'nearest'
        self.ref_window = 25

    @staticmethod
    def _project(x, y, prev, window):
        """与真实 builder 的同构逻辑：在 [prev-w, prev+w] 内找最近点，且不倒退。"""
        idx = torch.arange(91).float().view(1, -1)
        lo = prev.long().clamp(0, 90).view(-1, 1)
        hi = prev.long().clamp(0, 90).view(-1, 1)
        mask = (idx >= (lo - window).clamp(0, 90)) & (idx <= (hi + window).clamp(0, 90))
        d = (idx - x.view(-1, 1)) ** 2 + y.view(-1, 1) ** 2
        d = d.masked_fill(~mask, float('inf'))
        return torch.maximum(d.argmin(dim=1), prev.long())

    def get_ref_states_batch(self, w_i, x, y, ego_indices, p_i=None,
                             temporal_indices=None, ref_mode='index', window=25):
        B = len(w_i)
        t = temporal_indices.long().clamp(0, 90) if temporal_indices is not None \
            else torch.zeros(B, dtype=torch.long)
        if ref_mode == 'nearest':
            t = self._project(x, y, t, window)
        rx = t.float()
        return torch.stack([rx, torch.zeros(B), torch.full((B,), 10.0), torch.zeros(B),
                            torch.zeros(B), torch.zeros(B), t.float()], dim=-1)

    def get_road_dist_batch(self, w_i, temporal_indices, ego_indices, p_i, device):
        return torch.full((len(w_i),), 5.0, device=device)

    def get_expert_steer_batch(self, w_i, temporal_indices, ego_indices):
        return torch.zeros(len(w_i))


def make_config(**over):
    raw = yaml.safe_load(open(os.path.join(ROOT, 'src/configs/base.yaml'), encoding='utf-8'))
    merged = {}
    for k in ('training', 'agent', 'dynamics', 'world', 'diag'):
        merged.update(raw.get(k, {}) or {})
    merged.update(dict(buffer_capacity=512, batch_size=16, epochs=2,
                       num_worlds=4, pim_interval=3, horizon=6, window_size=8,
                       transformer_d_model=32, transformer_nhead=4,
                       transformer_num_layers=1, hidden_dim=32, init_penalty=0.01))
    merged.update(over)
    return SimpleNamespace(**merged)


def make_state(x=0.0, y=0.0, vlon=10.0, phi=0.0, t=0, n_others=0, lat1=0.0):
    s = np.zeros((1, TOTAL), dtype=np.float32)
    s[0, 0], s[0, 1], s[0, 2], s[0, 4] = x, y, vlon, phi
    for i in range(n_others):
        s[0, DIM_EGO + i * 4 + 0] = x + 4.0
        s[0, DIM_EGO + i * 4 + 1] = y
        s[0, DIM_EGO + i * 4 + 2] = 0.0
        s[0, DIM_EGO + i * 4 + 3] = 8.0
        s[0, 38 + i] = 1.0
    s[0, REF0 + 0] = 0.0      # dp
    s[0, REF0 + 1] = 0.0      # dphi
    s[0, REF0 + 2] = 0.0      # dv
    s[0, REF0 + 3] = lat1     # lat @ t+5
    s[0, REF0 + 4] = 0.0
    s[0, REF0 + 5] = 5.0
    s[0, REF0 + 6] = 10.0
    s[0, REF0 + 7] = lat1
    s[0, REF0 + 8] = 0.0
    s[0, REF0 + 9] = 5.0
    s[0, REF0 + 10] = 10.0
    s[0, REF0 + 11] = lat1
    s[0, REF0 + 12] = 0.0
    s[0, REF0 + 13] = 5.0
    s[0, REF0 + 14] = 10.0
    s[0, TEMP0] = t
    s[0, VEH0] = 5.286          # GPUDrive 用 size.length 当轴距（实测值之一）
    return torch.from_numpy(s)


def build_agent(cfg=None, W=4, n_others=0, **over):
    """over 里的键会覆盖 base.yaml 的配置项（如 road_in_penalty=False）。"""
    cfg = cfg or make_config(**over)
    env = SimpleNamespace(num_worlds=W, max_cont_agents=1)
    builder = FakeBuilder(W)
    agent = DiscreteIDCAgent(env, cfg, torch.device('cpu'), builder, [0] * W)
    return agent, builder, cfg


def check(name, cond, extra=''):
    print(('  PASS  ' if cond else '  FAIL  ') + name + (('  ' + extra) if extra else ''))
    if not cond:
        raise SystemExit(1)


print('== 1. 控制代价作用在物理量（饱和后不再增长） ==')
agent, builder, cfg = build_agent()
s0 = make_state()
w_i, p_i = [0], [0]
u_low = torch.tensor([[2.0, 0.0]])    # δ = 0.6 rad（刚好到上限）
u_high = torch.tensor([[6.0, 0.0]])   # 仍被 clamp 到 0.6 rad
l1 = agent.utility_batch(s0, u_low, w_i, p_i)
l2 = agent.utility_batch(s0, u_high, w_i, p_i)
check('raw 2.0 与 raw 6.0 的 utility 相同（δ 均已饱和）',
      float((l1 - l2).abs().max()) < 1e-6, f'|Δ|={float((l1-l2).abs().max()):.2e}')

u_a = torch.tensor([[0.0, 0.0]])
u_b = torch.tensor([[-2.0, 0.0]])
la = agent.utility_batch(s0, u_a, w_i, p_i)
lb = agent.utility_batch(s0, u_b, w_i, p_i)
check('raw -2.0 与 raw 0 的 δ 不同（-0.6 vs 0），utility 应不同',
      abs(float(la) - float(lb)) > 1e-3, f'Δ={abs(float(la)-float(lb)):.4f}')

print('== 2. validity 掩码真的生效 ==')
W = 1
agent2, b2, _ = build_agent(W=W)
s_none = make_state(n_others=0)
s_two = make_state(n_others=2)
# 把"幽灵车"摆到离自车 0.1m，验证未被 mask 时会炸、被 mask 时不动
s_none_bad = s_none.clone()
s_none_bad[0, DIM_EGO + 0] = 0.1
s_none_bad[0, DIM_EGO + 1] = 0.0
p_none = float(agent2.penalty_batch(s_none_bad, [0], [0]))
p_two = float(agent2.penalty_batch(s_two, [0], [0]))
check('validity=0 的车（极近）不产生惩罚', p_none < 1e-6, f'φ={p_none:.4f}')
check('validity=1 的车（4m）产生惩罚', p_two > 0.5, f'φ={p_two:.4f}')

print('== 3. 道路约束进 φ（penalty），不进 J_actor（utility） ==')
agent3, _, cfg3 = build_agent(W=1, n_others=0)
s_far = make_state(lat1=0.0)
s_off = make_state(lat1=4.5)          # 横向偏出 4.5m -> road_dist 5.0-4.5=0.5 < D_road_safe
u0 = torch.tensor([[0.0, 0.0]])
pen_far = float(agent3.penalty_batch(s_far, [0], [0]))
pen_off = float(agent3.penalty_batch(s_off, [0], [0]))
check('偏出道路时 φ 增大', pen_off > pen_far + 0.5, f'{pen_far:.3f} -> {pen_off:.3f}')

agent3b, _, _ = build_agent(W=1, n_others=0, road_in_penalty=False)
lu_far = float(agent3b.utility_batch(s_far, u0, [0], [0]))
lu_off = float(agent3b.utility_batch(s_off, u0, [0], [0]))
check('road_in_penalty=False 时退回软代价（utility 变大）', lu_off > lu_far, f'{lu_far:.3f} -> {lu_off:.3f}')

# 同一状态、只切 road_in_penalty 开关：差值应当正好等于被搬走的道路项
# ego_road_dist = 5.0 - 4.5 = 0.5 < D_road_safe 1.5 -> 旧软代价 = 1.0 * relu(1.0) = 1.0
l_on = float(agent3.utility_batch(s_off, u0, [0], [0]))
d_off = lu_off - l_on
l_on_far = float(agent3.utility_batch(s_far, u0, [0], [0]))
d_far = lu_far - l_on_far
check('road_in_penalty=True 时 utility 里没有道路项（偏出时差值为旧道路项）',
      abs(d_off - 1.0) < 1e-4, f'Δ={d_off:.4f}（期望 1.0）')
check('未偏出时两种开关下 utility 相同', abs(d_far) < 1e-4, f'Δ={d_far:.2e}')

print('== 4. 椭圆安全域分支可跑且方向正确 ==')
agent4, _, _ = build_agent(W=1, n_others=1, veh_safe_ellipse=True, ttc_weight=0.0)
s_side = make_state(n_others=1)
s_side[0, DIM_EGO + 0] = 0.0          # 同纵向位置
s_side[0, DIM_EGO + 1] = 1.0          # 侧向 1m（椭圆内）
s_side2 = s_side.clone()
s_side2[0, DIM_EGO + 1] = 4.0         # 侧向 4m（椭圆外）
p_in = float(agent4.penalty_batch(s_side, [0], [0]))
p_out = float(agent4.penalty_batch(s_side2, [0], [0]))
check('侧向 1m 有惩罚、4m 无惩罚', p_in > 0.1 and p_out < 1e-6, f'{p_in:.3f} / {p_out:.3f}')

print('== 5. f_pred_batch / rollout / critic / actor 全链路形状 ==')
agent5, _, _ = build_agent(W=4, n_others=2)
B = 8
# 让 ego 正好落在参考路径上（x = 索引，v=10 -> 每步前进 1 m = 1 个索引）
states = torch.cat([make_state(x=float(t % 88), n_others=2, t=t % 88) for t in range(B)], dim=0)
w5 = [0, 1, 2, 3] * 2
p5 = [0] * B
u5 = torch.zeros(B, 2)
sn = agent5.f_pred_batch(states, u5, w5, p5)
check(f'f_pred_batch 输出 {tuple(sn.shape)}', tuple(sn.shape) == (B, TOTAL))
check('temporal 索引前进一步（直线前进 1 m = 1 索引）',
      torch.all(sn[:, TEMP0] == states[:, TEMP0] + 1))

windows = states.unsqueeze(1).repeat(1, cfg.window_size, 1)   # [B, W, D]
tgt = agent5.compute_rollout_target(windows, w5, p5)
check(f'critic target 形状 {tuple(tgt.shape)}（无 NaN）', tuple(tgt.shape) == (B,))
check('critic target 无 NaN', not torch.isnan(tgt).any())

iw = torch.ones(B)
cl, td = agent5.update_critic(windows, w5, p5, iw)
check('update_critic 返回 (loss, 逐样本残差)', isinstance(cl, float) and td.shape == (B,),
      f'loss={cl:.3f}')

al, hv, mp = agent5.update_actor(windows, w5, p5, iw)
check('update_actor 返回 (loss, has_violation, max_penalty)',
      isinstance(al, float) and isinstance(hv, bool), f'aloss={al:.3f} max_p={mp:.3f}')

print('== 6. buffer 返回 IS 权重 + update() 端到端 + 优先级回写 ==')
agent6, _, cfg6 = build_agent(W=4, n_others=1)
for i in range(cfg6.batch_size + 4):
    st = make_state(n_others=1, t=i % 88, lat1=0.3 * (i % 5))
    win = np.tile(st.numpy(), (cfg6.window_size, 1)).astype(np.float32)
    agent6.buffer.handle_new_experience((win, i % 4, 0), priority=1.0 + i)
batch = agent6.buffer.sample_batch(8)
check('sample_batch 返回 4 项', len(batch) == 4)
w_np = batch[3]
check('IS 权重均值≈1', abs(float(w_np.mean()) - 1.0) < 1e-5, f'mean={w_np.mean():.4f}')
check('IS 权重非常数（确有校正）', float(w_np.std()) > 0, f'std={w_np.std():.4f}')

before = agent6.buffer.tree.tree.copy()
cl6, al6 = None, None
for _ in range(max(agent6.pev_interval, cfg6.pim_interval) + 1):
    cl6, al6 = agent6.update()
    if cl6 is not None:
        break
check('update() 在 PEV 命中时返回 critic_loss（其余步返回 None 是设计如此）',
      cl6 is not None, f'critic_loss={cl6}')
check('优先级已回写（树值发生变化）',
      not np.allclose(before, agent6.buffer.tree.tree))

print('== 7. penalty_schedule=gep 分支可跑 ==')
agent7, _, _ = build_agent(W=4, n_others=1, penalty_schedule='gep', gep_amplifier=1.5)
for i in range(cfg.batch_size + 4):
    st = make_state(n_others=1, t=i % 88)
    win = np.tile(st.numpy(), (cfg.window_size, 1)).astype(np.float32)
    agent7.buffer.handle_new_experience((win, i % 4, 0), priority=1.0)
r0 = agent7.rho
for _ in range(cfg.pim_interval * 2):
    agent7.update()
check('gep 调度下 rho 单调放大', agent7.rho > r0, f'{r0:.4f} -> {agent7.rho:.4f}')

print('== 8. 自车系编码：全局平移 + 旋转不变性 ==')
agent8, _, _ = build_agent(W=1, n_others=2, state_mode='ego')
s_a = make_state(x=3.0, y=-2.0, vlon=8.0, phi=0.3, n_others=2, t=10, lat1=0.4)


def rigid(st, theta, tx, ty):
    """对整个场景（自车 + 周车）施加刚体变换：先绕原点转 theta，再平移。"""
    out = st.clone()
    c, sn = np.cos(theta), np.sin(theta)
    out[0, 0] = float(st[0, 0]) * c - float(st[0, 1]) * sn + tx
    out[0, 1] = float(st[0, 0]) * sn + float(st[0, 1]) * c + ty
    out[0, 4] = float(st[0, 4]) + theta
    for i in range(8):
        b = DIM_EGO + i * 4
        ox, oy = float(st[0, b]), float(st[0, b + 1])
        out[0, b] = ox * c - oy * sn + tx
        out[0, b + 1] = ox * sn + oy * c + ty
        out[0, b + 2] = float(st[0, b + 2]) + theta
    return out


s_b = rigid(s_a, 0.7, 100.0, -50.0)
e_a, e_b = agent8._encode(s_a), agent8._encode(s_b)
check('编码对刚体变换不变（|Δ| < 1e-4）',
      float((e_a - e_b).abs().max()) < 1e-4, f'|Δ|={float((e_a - e_b).abs().max()):.2e}')
check('编码后自车位姿为 0（[0]=x,[1]=y,[3]=phi）',
      float(e_a[0, 0].abs() + e_a[0, 1].abs() + e_a[0, 3].abs()) < 1e-9)
check('编码确实抹掉了绝对位姿',
      abs(float(s_a[0, 0])) > 1.0 and abs(float(e_a[0, 0])) < 1e-9)
check(f'编码维度 = ENCODED_DIM({agent8.ENCODED_DIM})，丢掉 v_lat/omega',
      int(e_a.shape[-1]) == agent8.ENCODED_DIM
      and agent8.ENCODED_DIM == TOTAL - 2)
agent8b, _, _ = build_agent(W=1, n_others=2, state_mode='abs')
check('state_mode=abs 时保留绝对坐标（消融开关有效）',
      abs(float(agent8b._encode(s_a)[0, 0])) > 1.0)

print('== 9. 在线归一化 ==')
agent9, _, _ = build_agent(W=4, n_others=2)
agent9.input_norm = RunningNorm(agent9.ENCODED_DIM, prior_n=1.0)   # 收窄先验，便于断言
batch = torch.cat([make_state(x=float(i), y=float(-i % 7), vlon=5.0 + i % 11,
                              n_others=2, t=i % 88, lat1=0.1 * (i % 5))
                   for i in range(64)], dim=0)
enc = agent9._encode(batch)
for _ in range(50):
    agent9.input_norm.update(enc)
z = agent9._norm(enc)
varying = enc.std(dim=0) > 1e-6
check('归一化后各维均值≈0', float(z[:, varying].mean(dim=0).abs().max()) < 1e-3,
      f'max|mean|={float(z[:, varying].mean(dim=0).abs().max()):.2e}')
check('归一化后各维 std≈1', float((z[:, varying].std(dim=0) - 1).abs().max()) < 1e-2,
      f'max|std-1|={float((z[:, varying].std(dim=0) - 1).abs().max()):.2e}')
check('归一化输出被裁到 ±10 以内', float(z.abs().max()) <= 10.0 + 1e-6)
check('obs_norm=false 时跳过归一化',
      torch.allclose(build_agent(W=1, obs_norm=False)[0]._norm(enc), enc))

print('== 10. 参考点按位置推进（S6 回归测试） ==')
b10 = GPUDriveObservationBuilder.__new__(GPUDriveObservationBuilder)
idx10 = torch.arange(91).float()
b10.ref_tensor = torch.zeros(1, 1, 1, 91, 8)
b10.ref_tensor[0, 0, 0, :, 0] = idx10          # 直线，每索引 1 m
b10.ref_tensor[0, 0, 0, :, 2] = 10.0
b10.candidate_paths = {}


def proj(x, prev, mode='nearest'):
    r = b10.get_ref_states_batch([0], torch.tensor([x]), torch.tensor([0.0]),
                                 [0], [0], temporal_indices=torch.tensor([prev]),
                                 ref_mode=mode, window=25)
    return int(r[0, 6]), float(r[0, 0])


i0, x0 = proj(10.0, 5)
check('首次投影：从 prev=5 跳到最近点 10', i0 == 10 and abs(x0 - 10.0) < 1e-4, f'idx={i0}')
i1, _ = proj(10.0, i0)
check('ego 静止时索引不再前进（旧实现会一路 +1）', i1 == 10, f'idx={i1}')
i2, _ = proj(22.0, 10)
check('ego 前进 12 m，索引推进到 22', i2 == 22, f'idx={i2}')
i3, _ = proj(2.0, 22)
check('不倒退：最近点在后但索引保持 22', i3 == 22, f'idx={i3}')
i4, _ = proj(10.0, 7, mode='index')
check('ref_mode=index 保持旧行为（直接取 t）', i4 == 7, f'idx={i4}')

print('\n全部自检通过。')
