import math
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Tuple, List, Dict, Any

from models.continuous_actor_critic import (TransformerActor, ContinuousCritic, PaperMLPActor)
from models.kinematic_bicycle import KinematicBicycleModel
from buffer import PERBuffer
from utils import get_logger
from utils import save_checkpoint, load_checkpoint
from utils.running_norm import RunningNorm


def _cfgget(config, name, default):
    """先读顶层，再依次从 agent/training/dynamics/world 段里找（兼容嵌套配置）。"""
    v = getattr(config, name, None)
    if v is not None:
        return v
    for sec in ('agent', 'training', 'dynamics', 'world', 'diag', 'metrics'):
        obj = getattr(config, sec, None)
        if obj is None:
            continue
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    if isinstance(config, dict) and name in config:
        return config[name]
    return default



logger = get_logger('idc-agent')


# ==============================================
# IDC Agent
# ==============================================
class DiscreteIDCAgent:
    def __init__(self, env, config, device, state_builder, ego_indices):
        self.config = config
        self.env = env
        self.device = device
        self.num_worlds = env.num_worlds
        self.max_agents = env.max_cont_agents
        self.state_builder = state_builder
        self.ego_indices = ego_indices

        # IDC 维度定义
        self.DIM_EGO = 6
        self.DIM_OTHERS = 32
        self.DIM_VALIDITY = 8
        self.DIM_REF_ERROR = 18           # [dp, dphi, dv, lat+dphi+road+spd ×3(t+5/10/15) + curvahead+spdahead+rdahead]
        self.DIM_TEMPORAL = 1
        self.DIM_VEH = 1                  # 该车的 size.length（GPUDrive 拿它当轴距，逐车不同）
        self.TOTAL_STATE_DIM = (self.DIM_EGO + self.DIM_OTHERS + self.DIM_VALIDITY
                                + self.DIM_REF_ERROR + self.DIM_TEMPORAL + self.DIM_VEH)
        self.DIM_ROAD = 80

        # ---- 状态表示 / 参考点 / 归一化（决策 1 + 2）----
        # state_mode: 'ego' = 自车系表达（逐帧把自车放原点、航向 0，默认）
        #             'abs' = 旧行为，直接喂绝对坐标（消融用）
        self.state_mode = str(getattr(config, 'state_mode', 'ego'))
        # ref_mode:   'nearest' = 窗口内最近点投影，参考点随位置推进（默认）
        #             'index'   = 逐帧 +1，参考点随时钟推进（旧行为，消融用）
        self.ref_mode = str(getattr(config, 'ref_mode', 'nearest'))
        self.ref_window = int(getattr(config, 'ref_window', 25))
        self.obs_norm = bool(getattr(config, 'obs_norm', True))
        # 网络输入维度：ego 只保留 [x?, y?, v_lon, phi?]，丢掉 v_lat/omega
        # （GPUDrive 不提供这两个量，实测恒为 0，占着输入位纯属浪费）
        self.ENCODED_EGO = 4
        self.ENCODED_DIM = (self.ENCODED_EGO + self.DIM_OTHERS + self.DIM_VALIDITY
                            + self.DIM_REF_ERROR + self.DIM_TEMPORAL + self.DIM_VEH)
        self.input_norm = RunningNorm(self.ENCODED_DIM)
        # 把参考点模式同步给 state builder（obs 构建时也要按同一规则推进索引）
        if hasattr(state_builder, 'ref_mode'):
            state_builder.ref_mode = self.ref_mode
            state_builder.ref_window = self.ref_window

        # 超参
        self.dt = config.dt
        self.horizon = config.horizon
        self.batch_size = config.batch_size

        # 网络
        self.window_size = int(getattr(config, 'window_size', 16))
        self.actor_type = str(_cfgget(config, 'actor_type', 'transformer'))
        if self.actor_type == 'mlp':
            # 论文原版：单帧状态 + 2×256 ELU
            self.actor = PaperMLPActor(
                state_dim=self.ENCODED_DIM,
                hidden_dim=int(_cfgget(config, 'actor_hidden', 256)),
            ).to(device)
        else:
            self.actor = TransformerActor(
                state_dim=self.ENCODED_DIM,
                d_model=int(getattr(config, 'transformer_d_model', 256)),
                nhead=int(getattr(config, 'transformer_nhead', 4)),
                num_layers=int(getattr(config, 'transformer_num_layers', 2)),
                window_size=self.window_size,
                dropout=float(getattr(config, 'transformer_dropout', 0.1)),
            ).to(device)
        self.critic = ContinuousCritic(self.ENCODED_DIM, int(_cfgget(config, 'critic_hidden', 512))).to(device)
        self.dynamics = KinematicBicycleModel(
            dt=self.dt,
            L=config.wheelbase,
            lr_ratio=config.lr_ratio,
            v_max=config.v_max,
        )

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.lr_actor_max)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.lr_critic_max)

        # 余弦退火调度器
        # 【修正 S4】原来 actor/critic 共用 total_u_steps = 每 epoch 步数 × epochs，
        # 但 actor_scheduler 只在 PIM 里 step（每 pim_interval 个 env step 一次），
        # critic_scheduler 每个 env step 都 step。共用一个 T_max 会让 actor 的 LR
        # 全程只走完 cos 曲线的 1/pim_interval（约 3%），等于几乎不衰减。
        # 这里按"调度器实际被 step 的次数"分别标定。
        steps_per_epoch = int(getattr(state_builder, 'EXPERT_TRAJ_LEN', None) or 91)
        # 缓冲填满之前不训练，前 buffer_fill 个 env step 不产生 update()
        buffer_fill = max(1, int(math.ceil(self.batch_size / max(self.num_worlds, 1))))
        update_per_epoch = max(1, steps_per_epoch - buffer_fill)
        total_env_updates = update_per_epoch * config.epochs
        # 注意：self.pev_interval 在下面"目标函数一致性"块才定义，这里先就地取一次
        pev_iv = max(int(getattr(config, 'pev_interval', 1)), 1)
        total_pev_steps = max(1, total_env_updates // pev_iv)
        total_pim_steps = max(1, total_env_updates // max(config.pim_interval, 1))
        self.actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.actor_optimizer, T_max=total_pim_steps, eta_min=config.lr_actor_min)
        self.critic_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.critic_optimizer, T_max=total_pev_steps, eta_min=config.lr_critic_min)
        logger.info(f'[LR] steps/ep={steps_per_epoch} update/ep={update_per_epoch} '
                    f'T_max(actor)={total_pim_steps} T_max(critic)={total_pev_steps}')

        # IDC 成本权重
        self.pos_err_weight = config.pos_err_weight
        self.speed_err_weight = config.speed_err_weight
        self.heading_err_weight = config.heading_err_weight
        self.steer_cost_weight = config.steer_cost_weight
        self.acc_cost_weight = config.acc_cost_weight
        self.lookahead_pos_weight = config.lookahead_pos_weight
        self.lookahead_heading_weight = config.lookahead_heading_weight
        self.progress_weight = getattr(config, 'progress_weight', 0.0)
        self.ttc_weight = getattr(config, 'ttc_weight', 0.0)
        self.jerk_weight = getattr(config, 'jerk_weight', 0.0)
        self.jerk_max = getattr(config, 'jerk_max', 120.0)
        self.penalty_scale = getattr(config, 'penalty_scale', 1.0)
        self.road_weight = getattr(config, 'road_weight', 0.0)

        # GEP 惩罚
        self.rho = config.init_penalty
        self.max_penalty = config.max_penalty
        self.amplifier_c = config.amplifier_c
        self.gamma = config.gamma
        self.pev_step = 0
        self.pim_interval = config.pim_interval
        self._upd_step = 0

        # ---- 罚因子调度 ----
        # 'ema' : 现状（有界自适应，可升可降）
        # 'gep' : 论文 Algorithm 1 原版，每 m 次 PIM 后无条件 rho <- c*rho（单调放大）
        self.penalty_schedule = getattr(config, 'penalty_schedule', 'ema')
        self.gep_amplifier = float(getattr(config, 'gep_amplifier', 1.1))

        # ---- 目标函数一致性 ----
        # 截断作用在**物理偏差**上（而不是总代价）：超过阈值的偏差在 3 s 时域内
        # 本来就不可能弥补，继续加大只会污染梯度。utility_clamp 只作发散保护。
        self.pos_err_clamp = float(getattr(config, 'pos_err_clamp', 50.0))
        self.lat_err_clamp = float(getattr(config, 'lat_err_clamp', 20.0))
        self.utility_clamp = float(getattr(config, 'utility_clamp', 1.0e4))
        self.penalty_clamp = float(getattr(config, 'penalty_clamp', 100.0))
        # 训练节奏：PEV / PIM 各按 env step 计数
        self.pev_interval = max(int(getattr(config, 'pev_interval', 1)), 1)

        # ---- 采样权重 / 约束归属 / 安全域形状 ----
        self.use_is_weights = bool(getattr(config, 'use_is_weights', True))
        # True: 道路约束进 φ（论文 (1) 第三式是安全约束）；False: 退回 J_actor 软代价
        self.road_in_penalty = bool(getattr(config, 'road_in_penalty', True))
        # False: 双圆中心距标量阈值（现状）；True: 论文 (1) 的 Mahalanobis 椭圆
        self.veh_safe_ellipse = bool(getattr(config, 'veh_safe_ellipse', False))
        self.veh_safe_long = float(getattr(config, 'veh_safe_long', 5.0))
        self.veh_safe_lat = float(getattr(config, 'veh_safe_lat', 2.2))

        # 安全距离
        self.D_veh_safe = config.D_veh_safe
        self.D_road_safe = config.D_road_safe
        self.HALF_L = config.half_length
        self.HALF_W = config.half_width

        # 缓冲区
        self.buffer = PERBuffer(
            capacity=config.buffer_capacity,
            min_start_train=config.batch_size,
            window_size=self.window_size,
            state_dim=self.TOTAL_STATE_DIM,
        )
        self.global_step = 1
        # 每个 world 的"代际"：resample 换掉某些 world 时递增，缓冲区据此拒掉陈旧样本，
        # 这样其余 world 的经验不会被整体清空。
        self.world_gen = {w: 0 for w in range(self.num_worlds)}
        self.buffer.world_gen = self.world_gen

        # per-world 状态窗口（训练/推理时维护，不持久化）
        self.state_windows: Dict[int, Any] = {}
        from collections import deque
        self._deque = deque
        self.gep_iteration = 0

        # 道路状态缓存
        self.road_states = [None] * self.num_worlds

        # 探索噪声
        self.noise_std = config.noise_std
        self.noise_decay_rate = config.noise_decay_rate
        self.noise_std_min = config.noise_std_min

        # 诊断开关
        self.fix_speed = getattr(config, 'fix_speed', False)
        self.fix_heading = getattr(config, 'fix_heading', False)
        self.no_sign = getattr(config, 'no_sign', False)

        # 回合数
        self.globe_eps = 0

        # 历史损失
        self.history_loss = []
        self.epoch_history_pdms = []


    # ==========================================================
    # 动作映射：网络 raw 输出 -> 物理量（唯一实现）
    # select_action / f_pred_batch / utility_batch 必须共用同一份，
    # 否则"训练里用的动作"和"下发到环境的动作"会漂移（原实现抄了两份）。
    # ==========================================================
    STEER_SCALE = 0.3     # raw -> δ 的线性系数
    STEER_MAX = 0.6       # δ 上限 (rad)，与 GPUDrive 连续动作空间对齐
    ACC_MAX_POS = 1.5     # 正加速度上限 (m/s²)
    ACC_MAX_NEG = 3.0     # 制动上限 (m/s²)

    def _physical_action(self, u_raw: torch.Tensor):
        """u_raw: [..., 2] = [steer_raw, acc_raw] -> (a_phy, delta_phy)"""
        delta_phy = torch.clamp(u_raw[..., 0] * self.STEER_SCALE,
                                -self.STEER_MAX, self.STEER_MAX)
        acc_norm = torch.tanh(u_raw[..., 1])
        a_phy = torch.where(acc_norm >= 0,
                            acc_norm * self.ACC_MAX_POS,
                            acc_norm * self.ACC_MAX_NEG)
        return a_phy, delta_phy

    def _clamp_l(self, l: torch.Tensor) -> torch.Tensor:
        """单步 utility 截断。actor 与 critic 必须调用同一个函数。"""
        return torch.clamp(l, -self.utility_clamp, self.utility_clamp)

    def _clamp_p(self, p: torch.Tensor) -> torch.Tensor:
        return torch.clamp(p, max=self.penalty_clamp)

    # ==========================================================
    # 状态表示：绝对坐标 -> 自车系（逐帧）
    # ==========================================================
    def _encode(self, s: torch.Tensor) -> torch.Tensor:
        """绝对坐标状态 -> 网络特征（ENCODED_DIM 维）。

        - state_mode='ego'：逐帧把自车放原点、航向置 0，周车转成该帧自车系相对量
        - 丢掉 v_lat / omega（GPUDrive 不提供这两个量，实测恒为 0，占输入位无用）
        - validity / ref_error / 进度索引 / 车长 本来就与坐标系无关，原样保留

        s: [..., TOTAL_STATE_DIM] -> [..., ENCODED_DIM]
        """
        lead = s.shape[:-1]
        x, y, phi = s[..., 0], s[..., 1], s[..., 4]
        o = s[..., self.DIM_EGO:self.DIM_EGO + self.DIM_OTHERS].reshape(*lead, 8, 4)
        if self.state_mode == 'ego':
            c, sn = torch.cos(phi), torch.sin(phi)
            ex = torch.zeros_like(x)
            ey = torch.zeros_like(y)
            ephi = torch.zeros_like(phi)
            dx = o[..., 0] - x.unsqueeze(-1)
            dy = o[..., 1] - y.unsqueeze(-1)
            rx = dx * c.unsqueeze(-1) + dy * sn.unsqueeze(-1)
            ry = -dx * sn.unsqueeze(-1) + dy * c.unsqueeze(-1)
            dphi = o[..., 2] - phi.unsqueeze(-1)
            dphi = torch.atan2(torch.sin(dphi), torch.cos(dphi))
        else:
            ex, ey, ephi = x, y, phi
            rx, ry, dphi = o[..., 0], o[..., 1], o[..., 2]

        ego_enc = torch.stack([ex, ey, s[..., 2], ephi], dim=-1)      # 4 维
        others_enc = torch.stack([rx, ry, dphi, o[..., 3]], dim=-1).reshape(
            *lead, self.DIM_OTHERS)                                   # 32 维
        rest = s[..., self.DIM_EGO + self.DIM_OTHERS:]                # 28 维
        return torch.cat([ego_enc, others_enc, rest], dim=-1)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return self.input_norm(x) if self.obs_norm else x

    def _policy(self, s_window: torch.Tensor) -> torch.Tensor:
        """actor 的唯一入口：编码 -> 归一化 -> 前向。所有调用点都必须走这里。"""
        x = self._norm(self._encode(s_window))
        if self.actor_type == 'mlp':
            x = x[:, -1, :]        # 论文原版：只用当前帧
        return self.actor(x)

    def _value(self, s: torch.Tensor) -> torch.Tensor:
        """critic 的唯一入口。"""
        return self.critic(self._norm(self._encode(s)))

    def select_action(self, state, deterministic=False):
        """
        state: list of state objects (batch)，按世界索引排列
        返回 tensor [batch, max_agents, 2]，每个动作为 [acc, steer] (物理值)
        """
        with torch.no_grad():
            batch_windows = []
            for i, state_np in enumerate(state):
                if i not in self.state_windows:
                    self.state_windows[i] = self._deque(maxlen=self.window_size)
                self.state_windows[i].append(state_np)
                win_list = list(self.state_windows[i])
                window = np.tile(win_list[0], (self.window_size, 1)).astype(np.float32)
                window[-len(win_list):] = np.array(win_list)
                batch_windows.append(window)

            window_tensor = torch.from_numpy(
                np.stack(batch_windows)).to(self.device)
            enc_window = self._encode(window_tensor)
            # 用"策略实际访问到的状态分布"更新归一化统计（只取当前帧）
            self.input_norm.update(enc_window[:, -1, :])
            raw_output = self._policy(window_tensor)
            batch_size = raw_output.shape[0]
            raw_output = raw_output.view(batch_size, self.max_agents, 2)

            if not deterministic:
                noise = torch.normal(0, self.noise_std, size=raw_output.shape,
                                     device=raw_output.device)
                raw_output = raw_output + noise

            # raw -> 物理量（与 rollout / 代价计算共用同一实现）
            a_phy, delta_phy = self._physical_action(raw_output)
            action = torch.stack([a_phy, delta_phy], dim=-1)

            if self.global_step % 10 == 0:
                n_show = min(20, batch_size)
                for i in range(n_show):
                    logger.debug(f'[DIAG-act] world_{i} acc={action[i,0,0].item():.3f} '
                                f'steer={action[i,0,1].item():.3f}')
            return action

    def reset_world_state(self, world_idx):
        self.state_windows.pop(world_idx, None)


    def predict_others(self,others_state):
        """
        others_state: [batch, N * 4]  (px, py, phi, speed)
        返回下一时刻的 others_state
        """
        logger.debug(f'预测周车下一状态: others_state shape={others_state.shape}')
        px, py, phi, speed = others_state.view(others_state.shape[0], int(self.DIM_OTHERS / 4), 4).unbind(dim=-1)  # [batch, N]
        px_next = px + speed * torch.cos(phi) * self.dt
        py_next = py + speed * torch.sin(phi) * self.dt
        phi_next = phi
        speed_next = speed
        return torch.stack([px_next, py_next, phi_next, speed_next], dim=-1)
    
    def compute_rollout_target(self, windows, world_indices, path_indices=None):
        total_utility = torch.zeros(windows.shape[0], device=self.device)
        s_window = windows
        l_min_log, l_max_log = float('inf'), float('-inf')
        for t in range(self.horizon):
            u = self._policy(s_window)
            if torch.isnan(u).any():
                logger.warning(f'[NaN] actor output at rollout step {t}')
            s_next = self.f_pred_batch(s_window[:, -1, :], u, world_indices, path_indices)
            if torch.isnan(s_next).any():
                logger.warning(f'[NaN] f_pred state at rollout step {t}')
            l = self.utility_batch(s_next, u, world_indices, path_indices)
            if torch.isnan(l).any():
                logger.warning(f'[NaN] utility at rollout step {t}')
            l_min_log = min(l_min_log, l.min().item())
            l_max_log = max(l_max_log, l.max().item())
            total_utility = total_utility + (self.gamma ** t) * self._clamp_l(l)
            s_window = torch.cat([s_window[:, 1:, :], s_next.unsqueeze(1)], dim=1)
        if self.global_step % 50 == 0:
            tu = total_utility
            logger.debug(f'[DIAG-critic] utility raw range=[{l_min_log:.2f}, {l_max_log:.2f}], '
                        f'target raw mean={tu.mean().item():.2f} min={tu.min().item():.2f} max={tu.max().item():.2f}')
        return total_utility

    def f_pred_batch(self, states, actions, w_i, p_i=None):
        """
        states: tensor [batch, TOTAL_STATE_DIM]
        actions: tensor [batch, 2]
        w_i: list/tuple of world indices [batch]
        p_i: optional list of path indices [batch]
        """
        ego_tensors = states[:, :self.DIM_EGO]
        others_tensors = states[:, self.DIM_EGO:self.DIM_EGO+self.DIM_OTHERS]
        # 提取 validity mask 并在推演中保持不变
        val_start = self.DIM_EGO + self.DIM_OTHERS
        validity_tensors = states[:, val_start:val_start + self.DIM_VALIDITY]
        # 路径进度索引（nearest 模式下会被"投影到最近点"更新；index 模式下逐帧 +1）
        temp_start = self.DIM_EGO + self.DIM_OTHERS + self.DIM_VALIDITY + self.DIM_REF_ERROR
        temporal_idx = states[:, temp_start:temp_start + self.DIM_TEMPORAL].squeeze(-1).long()
        
        # 1. 动力学推演（raw -> 物理量，与 select_action 共用同一实现）
        a_phy, delta_phy = self._physical_action(actions)
        actions_phy = torch.stack([a_phy, delta_phy], dim=-1)  # [acc, steer]
        # 轴距逐车取：GPUDrive 用 size.length 当轴距，逐车不同（实测 4.80~5.29 m）
        veh_len = states[:, self.TOTAL_STATE_DIM - 1:self.TOTAL_STATE_DIM]
        ego_next = self.dynamics(ego_tensors[..., :6], actions_phy, wheelbase=veh_len)
        x_raw, y_raw, theta_next, v_next = ego_next[:, 0], ego_next[:, 1], ego_next[:, 2], ego_next[:, 3]

        # 构建 world_t, path_t 用于 ref_tensor 直接索引
        wt = torch.tensor(list(w_i), device=x_raw.device, dtype=torch.long)
        pt = torch.tensor(list(p_i), device=x_raw.device, dtype=torch.long) if p_i is not None else torch.zeros_like(wt)

        # 2. 参考点查询
        #    nearest: 在上一进度索引的窗口内做最近点投影 -> 参考点随"位置"推进，
        #             不会因为 ego 慢下来就跑到前面去（修 S6）
        #    index  : 旧行为，逐帧 +1（参考点随时钟推进）
        refs = self.state_builder.get_ref_states_batch(
            w_i, x_raw.detach(), y_raw.detach(), self.ego_indices, p_i,
            temporal_indices=temporal_idx, ref_mode=self.ref_mode,
            window=self.ref_window)
        ref_x, ref_y = refs[:, 0], refs[:, 1]
        if self.ref_mode == 'nearest':
            temporal_next = torch.clamp(refs[:, 6].long(), max=90)
        else:
            temporal_next = (temporal_idx + 1).long()

        # 3. 直接使用动力学输出（坏world已由训练脚本过滤，不再需要clamp）
        x_next = x_raw
        y_next = y_raw

        # 4. 组装 next_state
        ego_next_formatted = torch.stack([
            x_next, y_next, 
            v_next, torch.zeros_like(v_next),  # v_lon, v_lat
            theta_next, torch.zeros_like(theta_next) # phi, omega
        ], dim=-1)
        
        others_next = self.predict_others(others_tensors)
        
        # 5. 参考误差（在动力学输出上计算）
        dx = ref_x - x_next
        dy = ref_y - y_next
        delta_p = torch.hypot(dx, dy)
        if not self.no_sign:
            cross = dy * torch.cos(refs[:, 4]) - dx * torch.sin(refs[:, 4])
            sign = torch.where(cross >= 0, torch.tensor(1.0, device=self.device),
                               torch.tensor(-1.0, device=self.device))
            delta_p = delta_p * sign

        # 诊断：clamp 后仍大偏离
        if (delta_p.abs() > 100.0).any():
            big = (delta_p.abs() > 100.0).nonzero(as_tuple=True)[0].tolist()
            for i in big[:3]:
                logger.debug(f'[FPRED-ERR] sample_{i} delta_p={delta_p[i].item():.1f}m '
                               f'ego=({x_next[i].item():.1f},{y_next[i].item():.1f}) '
                               f'ref=({ref_x[i].item():.1f},{ref_y[i].item():.1f})')

        delta_phi = refs[:, 4] - theta_next
        delta_phi = torch.atan2(torch.sin(delta_phi), torch.cos(delta_phi))
        
        ego_speed = torch.hypot(ego_next_formatted[:, 2], ego_next_formatted[:, 3])
        delta_v = ego_speed - refs[:, 2]

        # 6. 前瞻参考点：给 Actor 前方弯道/道路信息（t+5, t+10, t+15）
        tl1 = torch.clamp(temporal_next + 5, max=90)
        tl2 = torch.clamp(temporal_next + 10, max=90)
        tl3 = torch.clamp(temporal_next + 15, max=90)
        refs_l1 = self.state_builder.get_ref_states_batch(
            w_i, x_raw.detach(), y_raw.detach(), self.ego_indices, p_i,
            temporal_indices=tl1, ref_mode='index')
        refs_l2 = self.state_builder.get_ref_states_batch(
            w_i, x_raw.detach(), y_raw.detach(), self.ego_indices, p_i,
            temporal_indices=tl2, ref_mode='index')
        refs_l3 = self.state_builder.get_ref_states_batch(
            w_i, x_raw.detach(), y_raw.detach(), self.ego_indices, p_i,
            temporal_indices=tl3, ref_mode='index')

        lat_l1 = (refs_l1[:, 1] - y_next) * torch.cos(refs_l1[:, 4]) - (refs_l1[:, 0] - x_next) * torch.sin(refs_l1[:, 4])
        dphi_l1 = refs_l1[:, 4] - theta_next
        dphi_l1 = torch.atan2(torch.sin(dphi_l1), torch.cos(dphi_l1))
        road_l1 = self.state_builder.get_road_dist_batch(w_i, tl1, self.ego_indices, p_i, x_next.device)
        spd_l1 = refs_l1[:, 2]

        lat_l2 = (refs_l2[:, 1] - y_next) * torch.cos(refs_l2[:, 4]) - (refs_l2[:, 0] - x_next) * torch.sin(refs_l2[:, 4])
        dphi_l2 = refs_l2[:, 4] - theta_next
        dphi_l2 = torch.atan2(torch.sin(dphi_l2), torch.cos(dphi_l2))
        road_l2 = self.state_builder.get_road_dist_batch(w_i, tl2, self.ego_indices, p_i, x_next.device)
        spd_l2 = refs_l2[:, 2]

        lat_l3 = (refs_l3[:, 1] - y_next) * torch.cos(refs_l3[:, 4]) - (refs_l3[:, 0] - x_next) * torch.sin(refs_l3[:, 4])
        dphi_l3 = refs_l3[:, 4] - theta_next
        dphi_l3 = torch.atan2(torch.sin(dphi_l3), torch.cos(dphi_l3))
        road_l3 = self.state_builder.get_road_dist_batch(w_i, tl3, self.ego_indices, p_i, x_next.device)
        spd_l3 = refs_l3[:, 2]

        # 指数衰减前视特征（从 ref_tensor 直接读取）
        tn = torch.clamp(temporal_next, max=90)
        curv_ahead = self.state_builder.ref_tensor[wt, 0, pt, tn, 5]
        spd_ahead = self.state_builder.ref_tensor[wt, 0, pt, tn, 6]
        road_ahead = self.state_builder.ref_tensor[wt, 0, pt, tn, 7]

        ref_error_tensors = torch.stack([delta_p, delta_phi, delta_v,
                                          lat_l1, dphi_l1, road_l1, spd_l1,
                                          lat_l2, dphi_l2, road_l2, spd_l2,
                                          lat_l3, dphi_l3, road_l3, spd_l3,
                                          curv_ahead, spd_ahead, road_ahead], dim=-1)
        
        # 组合并返回最终状态
        return torch.cat([ego_next_formatted, others_next.view(len(states), -1),
                          validity_tensors, ref_error_tensors,
                          temporal_next.unsqueeze(-1).float(), veh_len], dim=-1)
    
    def update_critic(self, windows, world_indices, path_indices=None, is_weights=None):
        logger.debug(f'更新 Critic: windows shape={windows.shape}')
        with torch.no_grad():
            targets = self.compute_rollout_target(windows, world_indices, path_indices).detach()
        logger.debug(f"[DEBUG] target stats: min={targets.min().item():.2f}, max={targets.max().item():.2f}, mean={targets.mean().item():.2f}, has_nan={torch.isnan(targets).any()}")

        if torch.isnan(targets).any():
            logger.warning(f'Target 包含 NaN，跳过本次 Critic 更新')
            return float('nan'), None

        values = self._value(windows[:, -1, :])
        logger.debug(f"[DEBUG] value stats: min={values.min().item():.2f}, max={values.max().item():.2f}, mean={values.mean().item():.2f}, has_nan={torch.isnan(values).any()}")

        # 逐样本残差：既用于 MSE，也用于 PER 优先级回写（只做优先采样不做偏差
        # 校正会引入估计偏差，见 per_buffer.sample_batch 的 IS 权重）
        with torch.no_grad():
            td_abs = (targets.unsqueeze(1) - values).abs().view(-1)
        loss_per = (values - targets.unsqueeze(1)) ** 2
        if is_weights is not None:
            loss = (is_weights * loss_per.view(-1)).mean()
        else:
            loss = loss_per.mean()
        logger.debug(f"[DEBUG] loss = {loss.item():.4f}")

        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        # 打印梯度范数
        total_norm = 0
        for p in self.critic.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        logger.debug(f"[DEBUG] grad norm = {total_norm:.4f}")

        self.critic_optimizer.step()
        self.critic_scheduler.step()
        return loss.detach().item(), td_abs.cpu().numpy()

    def update_actor(self, windows, w_i, p_i=None, is_weights=None):
        total_cost = torch.zeros(windows.shape[0], device=self.device)
        s_window = windows
        ref_start = self.DIM_EGO + self.DIM_OTHERS + self.DIM_VALIDITY
        max_penalty = 0.0
        prev_raw_acc = torch.zeros(windows.shape[0], device=self.device)
        for t in range(self.horizon):
            u = self._policy(s_window)
            s_next = self.f_pred_batch(s_window[:, -1, :], u, w_i, p_i)
            l = self.utility_batch(s_next, u, w_i, p_i)

            # Jerk 惩罚：抑制加速度高频跳动
            if self.jerk_weight > 0 and t > 0:
                curr_raw_acc = torch.tanh(u[:, 1])
                jerk = (curr_raw_acc - prev_raw_acc) / self.dt
                l = l + self.jerk_weight * (jerk / self.jerk_max) ** 2
            prev_raw_acc = torch.tanh(u[:, 1])
            if t == 0:
                logger.info(f'[LA-DIAG] l1(lat={s_next[0, ref_start+3]:.2f} dphi={s_next[0, ref_start+4]:.3f} road={s_next[0, ref_start+5]:.1f} spd={s_next[0, ref_start+6]:.1f}) '
                            f'l2(lat={s_next[0, ref_start+7]:.2f} dphi={s_next[0, ref_start+8]:.3f} road={s_next[0, ref_start+9]:.1f} spd={s_next[0, ref_start+10]:.1f}) '
                            f'l3(lat={s_next[0, ref_start+11]:.2f} dphi={s_next[0, ref_start+12]:.3f} road={s_next[0, ref_start+13]:.1f} spd={s_next[0, ref_start+14]:.1f}) '
                            f'delta_p={s_next[0, ref_start]:.2f} delta_phi={s_next[0, ref_start+1]:.3f} spd={s_next[0, ref_start+2]:.1f}')
            l = self._clamp_l(l)
            p = self._clamp_p(self.penalty_batch(s_next, w_i, p_i))
            if not torch.isfinite(l).all() or not torch.isfinite(p).all():
                logger.warning(
                    f'[NaN-src] t={t} u={bool(torch.isfinite(u).all())} '
                    f's_next={bool(torch.isfinite(s_next).all())} '
                    f'l={bool(torch.isfinite(l).all())} p={bool(torch.isfinite(p).all())} '
                    f'ego={bool(torch.isfinite(s_next[:, :6]).all())} '
                    f'oth={bool(torch.isfinite(s_next[:, 6:38]).all())} '
                    f'ref={bool(torch.isfinite(s_next[:, 46:64]).all())}')
            max_penalty = max(max_penalty, p.max().item())
            if t == 0:
                logger.debug(f'[DIAG-pen] raw penalty min={p.min().item():.2f} max={p.max().item():.2f} '
                            f'mean={p.mean().item():.2f}  (rho={self.rho:.4f})')
            total_cost = total_cost + (self.gamma ** t) * (l + self.rho * p)
            s_window = torch.cat([s_window[:, 1:, :], s_next.unsqueeze(1)], dim=1)

        if is_weights is not None:
            actor_loss = (is_weights * total_cost).mean()
        else:
            actor_loss = total_cost.mean()
        has_violation = max_penalty > 0.5

        bc_weight = getattr(self.config, 'bc_weight', 0.0)
        if bc_weight > 0:
            temp_start = self.DIM_EGO + self.DIM_OTHERS + self.DIM_VALIDITY + self.DIM_REF_ERROR
            temporal_idx = windows[:, -1, temp_start:temp_start + self.DIM_TEMPORAL].squeeze(-1).long()
            expert_steer = self.state_builder.get_expert_steer_batch(
                w_i, temporal_idx, self.ego_indices)
            raw_init = self._policy(windows)
            _, actor_steer = self._physical_action(raw_init)
            bc_loss = ((actor_steer - expert_steer.to(actor_steer.device)) ** 2).mean()
            if not torch.isfinite(bc_loss):
                logger.warning(f'[NaN-src] bc_loss 非有限: actor_steer_finite='
                               f'{bool(torch.isfinite(actor_steer).all())} '
                               f'expert_finite={bool(torch.isfinite(expert_steer).all())}')
            _bc_scale = actor_loss.detach().abs() / (bc_loss.detach() + 1e-8)
            actor_loss = actor_loss + bc_weight * bc_loss * _bc_scale
            if self.global_step % 50 == 0:
                logger.info(f'[DIAG-bc] bc_loss={bc_loss.item():.4f} weight={bc_weight} scaled={float(bc_weight*_bc_scale):.4f} '
                            f'expert_mean={expert_steer.mean().item():.3f}')
        else:
            bc_loss = None

        # 衰减探索噪声（所有模式统一衰减）
        old_std = self.noise_std
        self.noise_std = max(self.noise_std_min, self.noise_std * self.noise_decay_rate)
        if abs(old_std - self.noise_std) > 1e-4:
            logger.debug(f'[DIAG-noise] decay {old_std:.4f} -> {self.noise_std:.4f}')

        # 保存权重副本，用于 NaN 回滚
        backup = {name: param.clone() for name, param in self.actor.named_parameters()}

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        self.actor_scheduler.step()

        # 检测 NaN 权重并回滚
        has_nan = False
        for name, param in self.actor.named_parameters():
            if torch.isnan(param).any():
                has_nan = True
                break
        if has_nan:
            logger.warning(f'Actor 权重出现 NaN，回滚到更新前。rho={self.rho:.4f}, gep_iter={self.gep_iteration}')
            with torch.no_grad():
                for name, param in self.actor.named_parameters():
                    param.copy_(backup[name])
            # rho 是乘性罚因子，回滚时保持不动即可。
            # （原实现先 pev_step -= 1、再 gep_iteration -= 1、再 rho -= amplifier_c，
            #  三者语义都不成立：pev_step 在进函数前已被清零，rho 是倍数不是增量。）
            actor_loss = float('nan')
            has_violation = False
            max_penalty = 0.0

        return actor_loss.detach().item() if not has_nan else float('nan'), has_violation, max_penalty

    def utility_batch(self, s, u, w_i, p_i=None):
        """
        s: [batch, TOTAL_STATE_DIM], u: [batch, 2]
        """
        ego = s[:, :self.DIM_EGO]
        # 用时序索引替代空间最近点查表
        temp_start = self.DIM_EGO + self.DIM_OTHERS + self.DIM_VALIDITY + self.DIM_REF_ERROR
        temporal_idx = s[:, temp_start:temp_start + self.DIM_TEMPORAL].squeeze(-1).long()
        refs = self.state_builder.get_ref_states_batch(
            w_i, ego[:, 0], ego[:, 1], self.ego_indices, p_i,
            temporal_indices=temporal_idx)

        pos_err = torch.sqrt((ego[:, 0] - refs[:, 0]) ** 2
                            + (ego[:, 1] - refs[:, 1]) ** 2 + 1e-8)
        pos_err = torch.clamp(pos_err, max=self.pos_err_clamp)
        # 直接读 state 中的 delta_phi（与 f_pred_batch 和 _calc_ref_error 含义一致）
        ref_start = self.DIM_EGO + self.DIM_OTHERS + self.DIM_VALIDITY
        heading_err = torch.clamp(s[:, ref_start + 1], -math.pi, math.pi)
        speed_err = torch.sqrt((ego[:, 2] - refs[:, 2]) ** 2
                              + (ego[:, 3] - refs[:, 3]) ** 2 + 1e-8)
        speed_err = torch.clamp(speed_err, max=30.0)

        # 前瞻横向偏差同样按物理量截断
        lat1 = torch.clamp(s[:, ref_start + 3], -self.lat_err_clamp, self.lat_err_clamp)
        lat2 = torch.clamp(s[:, ref_start + 7], -self.lat_err_clamp, self.lat_err_clamp)
        lat3 = torch.clamp(s[:, ref_start + 11], -self.lat_err_clamp, self.lat_err_clamp)

        # 控制代价必须作用在物理量 δ(rad) / a(m/s²) 上（论文 R = diag(0.1, 0.005)）。
        # 不能用网络 raw 输出：raw 在动作饱和后仍无界增长，会把"已经饱和的动作"
        # 继续往 0 推，制造 steer bang-bang（满舵时原实现约比论文重 17 倍）。
        a_phy, delta_phy = self._physical_action(u)
        steer_cost = delta_phy ** 2
        acc_cost = a_phy ** 2

        l = (self.pos_err_weight * pos_err ** 2
             + (0.0 if self.fix_speed else self.speed_err_weight) * speed_err ** 2
             + (0.0 if self.fix_heading else self.heading_err_weight) * heading_err ** 2
             + self.steer_cost_weight * steer_cost
             + self.acc_cost_weight * acc_cost
             + self.lookahead_pos_weight * (lat1 ** 2 + lat2 ** 2 + lat3 ** 2)
             + self.lookahead_heading_weight * (s[:, ref_start + 4] ** 2 + s[:, ref_start + 8] ** 2 + s[:, ref_start + 12] ** 2))

        if self.progress_weight > 0:
            speed_along_path = ego[:, 2] * torch.cos(s[:, ref_start + 1])
            progress = torch.clamp(speed_along_path, min=0.0) * self.dt
            l = l - self.progress_weight * progress

        # 道路约束默认已移到 penalty_batch（论文 (1) 把路沿当安全约束，需要被 rho
        # 放大）。这里保留 road_in_penalty=False 时的旧软代价路径，便于做消融。
        if (not self.road_in_penalty) and self.road_weight > 0 \
                and not getattr(self.config, 'no_road_penalty', False):
            lat = s[:, ref_start + 3]
            road_dist_ref = torch.clamp(self.state_builder.get_road_dist_batch(
                w_i, temporal_idx, self.ego_indices,
                p_i if p_i is not None else [0]*len(w_i), s.device), max=50.0)
            ego_road_dist = torch.clamp(road_dist_ref - torch.abs(lat), min=0.0)
            l = l + self.road_weight * F.relu(self.D_road_safe - ego_road_dist)

        # 诊断：每次调用自动抓 pos_err 最大的样本
        max_idx = pos_err.argmax().item()
        if pos_err[max_idx] > 5.0:
            logger.debug(f'[TRACK-DIAG] worst sample_{max_idx} world_{w_i[max_idx]} '
                        f'ego=({ego[max_idx,0]:.1f},{ego[max_idx,1]:.1f},θ={ego[max_idx,4]:.2f}) '
                        f'ref=({refs[max_idx,0]:.1f},{refs[max_idx,1]:.1f},θ={refs[max_idx,4]:.2f}) '
                        f'nearest={int(refs[max_idx,6].item())} pos={pos_err[max_idx]:.1f} dhead={heading_err[max_idx]:.2f}')
        return l

    def _two_circle_centers(self, x, y, phi):
        """
        x, y, phi: [batch, ...] tensors
        返回前圆心和后圆心: (front, rear) 各 [batch, ..., 2]
        """
        dx = self.HALF_L * torch.cos(phi)
        dy = self.HALF_L * torch.sin(phi)
        front = torch.stack([x + dx, y + dy], dim=-1)
        rear = torch.stack([x - dx, y - dy], dim=-1)
        return front, rear

    def penalty_batch(self, s, w_i, p_i=None):
        """
        s: [batch, TOTAL_STATE_DIM], p_i: path indices [batch]
        返回 [batch] 标量惩罚值（双圆碰撞模型 + 道路边界，可导）
        """
        ego = s[:, :self.DIM_EGO]
        others = s[:, self.DIM_EGO:self.DIM_EGO + self.DIM_OTHERS].view(-1, 8, 4)
        val_start = self.DIM_EGO + self.DIM_OTHERS
        validity = s[:, val_start:val_start + self.DIM_VALIDITY]  # [batch, 8]

        # 自车双圆心 [batch, 2, 2]
        ego_front, ego_rear = self._two_circle_centers(ego[:, 0], ego[:, 1], ego[:, 4])
        ego_circles = torch.stack([ego_front, ego_rear], dim=1)  # [batch, 2, 2]

        # 周车双圆心 [batch, 8, 2, 2]
        oth_front, oth_rear = self._two_circle_centers(
            others[:, :, 0], others[:, :, 1], others[:, :, 2])
        oth_circles = torch.stack([oth_front, oth_rear], dim=2)  # [batch, 8, 2, 2]

        # 4 组距离: ego[2] × other[8,2] → [batch, 8, 2, 2]
        diff = ego_circles[:, None, :, None, :] - oth_circles[:, :, None, :, :]
        dist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)  # [batch, 8, 2, 2]

        if self.veh_safe_ellipse:
            # 论文 (1) 的 Mahalanobis 形式：(x-xj)ᵀ M (x-xj) >= D_safe。
            # 把圆心差投影到周车自身坐标系，纵横半轴分别归一化 -> 椭圆安全域，
            # 解决"单标量阈值在纵向过保守、在横向贴着车道线就报警"的问题。
            cphi = torch.cos(others[:, :, 2]).unsqueeze(2).unsqueeze(2)   # [batch, 8, 1, 1]
            sphi = torch.sin(others[:, :, 2]).unsqueeze(2).unsqueeze(2)
            dx, dy = diff[..., 0], diff[..., 1]
            long_n = (dx * cphi + dy * sphi) / max(self.veh_safe_long, 1e-6)
            lat_n = (-dx * sphi + dy * cphi) / max(self.veh_safe_lat, 1e-6)
            norm_dist = torch.sqrt(long_n ** 2 + lat_n ** 2 + 1e-8)  # =1 即贴在安全椭圆上
            pair_pen = F.relu(1.0 - norm_dist)                        # 一次方，与现状同阶
        else:
            pair_pen = F.relu(self.D_veh_safe - dist)                # 一次方：边界附近连续梯度

        per_veh_max = pair_pen.flatten(2).max(dim=2).values       # [batch, 8]
        veh_violation = (per_veh_max * validity).sum(dim=1)       # [batch]

        # TTC 连续惩罚：给碰撞前的逼近-减速梯度
        ttc_violation = torch.zeros_like(veh_violation)
        if self.ttc_weight > 0:
            ego_v = ego[:, 2]
            ego_h = ego[:, 4]
            rel_x = others[:, :, 0] - ego[:, 0].unsqueeze(1)
            rel_y = others[:, :, 1] - ego[:, 1].unsqueeze(1)
            ttc_dist = torch.sqrt(rel_x ** 2 + rel_y ** 2 + 1e-8)
            fwd_x = torch.cos(ego_h).unsqueeze(1)
            fwd_y = torch.sin(ego_h).unsqueeze(1)
            fwd_align = (rel_x * fwd_x + rel_y * fwd_y) / (ttc_dist + 1e-8)
            other_v = others[:, :, 3]
            ttc_closing = torch.clamp(ego_v.unsqueeze(1) - other_v * fwd_align, min=0.0)
            ttc = ttc_dist / (ttc_closing + 1e-4)
            ttc_pen = torch.clamp(4.0 - ttc, min=0.0, max=4.0) / 4.0
            ttc_violation = (ttc_pen * validity).sum(dim=1) * self.ttc_weight

        if getattr(self.config, 'no_veh_penalty', False):
            veh_violation = torch.zeros_like(veh_violation)

        # ---- 道路边界约束（论文 (1) 第三式，属于安全约束 -> 进 φ 被 rho 放大）----
        road_violation = torch.zeros_like(veh_violation)
        if self.road_in_penalty and not getattr(self.config, 'no_road_penalty', False):
            ref_start = self.DIM_EGO + self.DIM_OTHERS + self.DIM_VALIDITY
            temp_start = ref_start + self.DIM_REF_ERROR
            temporal_idx = s[:, temp_start:temp_start + self.DIM_TEMPORAL].squeeze(-1).long()
            lat = s[:, ref_start + 3]                       # 前瞻点 (t+5) 的横向偏移
            road_dist_ref = torch.clamp(self.state_builder.get_road_dist_batch(
                w_i, temporal_idx, self.ego_indices,
                p_i if p_i is not None else [0] * len(w_i), s.device), max=50.0)
            ego_road_dist = torch.clamp(road_dist_ref - lat.abs(), min=0.0)
            road_violation = F.relu(self.D_road_safe - ego_road_dist) * self.road_weight

        return veh_violation + ttc_violation + road_violation
    
    def update(self):
        # PEV / PIM 各自按 env step 计数：原来 critic 每步都算一次 30 步 rollout，
        # 而 critic 目前没有任何消费者，等于白烧绝大部分算力。
        self._upd_step += 1
        do_pev = (self._upd_step % self.pev_interval == 0)
        do_pim = (self._upd_step % max(self.pim_interval, 1) == 0)
        if not (do_pev or do_pim):
            return None, None

        windows, word_indexs, path_indices, is_weights_np = self.buffer.sample_batch(self.batch_size)
        if len(windows) == 0:
            return None, None
        windows_tensor = torch.from_numpy(windows).to(self.device)
        # 兜底：坏场景的哨兵值会让整条 rollout 变 NaN，先清一遍
        bad = ~torch.isfinite(windows_tensor)
        if bad.any():
            n_bad = int(bad.any(dim=(1, 2)).sum())
            logger.warning(f'[NaN-guard] {n_bad}/{len(windows)} 个样本含 inf/nan，已置零')
            windows_tensor = torch.nan_to_num(windows_tensor, nan=0.0,
                                              posinf=0.0, neginf=0.0)
        if self.use_is_weights:
            is_weights = torch.from_numpy(is_weights_np).to(self.device).float()
        else:
            is_weights = None

        # PEV（策略评估）
        critic_loss, td_abs = None, None
        if do_pev:
            critic_loss, td_abs = self.update_critic(
                windows_tensor, word_indexs, path_indices, is_weights)

        # PIM（策略改进）
        actor_loss = None
        if do_pim:
            actor_loss, has_violation, max_p = self.update_actor(
                windows_tensor, word_indexs, path_indices, is_weights)
            self.gep_iteration += 1

            if self.penalty_schedule == 'gep':
                # 论文 Algorithm 1：每 m 次策略改进后无条件 rho <- c*rho（单调放大）
                old_rho = self.rho
                self.rho = min(self.rho * self.gep_amplifier, self.max_penalty)
                logger.info(f'[GEP-gep] rho {old_rho:.4f} -> {self.rho:.4f} '
                            f'gep={self.gep_iteration} max_p={max_p:.2f}')
            elif has_violation:
                target = self.config.init_penalty + min(max_p, 50.0) * 0.02
                self.rho = min(0.95 * self.rho + 0.05 * target, self.max_penalty)
                logger.info(f'[GEP-ema] violation, max_p={max_p:.0f} target={target:.3f} '
                            f'rho={self.rho:.4f} gep={self.gep_iteration}')
            else:
                self.rho = max(0.99 * self.rho + 0.01 * self.config.init_penalty,
                               self.config.init_penalty)
                logger.info(f'[GEP-ema] clean PIM, rho={self.rho:.4f} gep={self.gep_iteration}')

        # PER 优先级回写（原实现从未调用，优先级只在插入时按跟踪误差定一次）
        if td_abs is not None:
            self.buffer.update_last_batch_priorities(td_abs)

        return critic_loss, actor_loss
        
    def update_ego_indices(self, new_ego_indices):
        self.ego_indices = new_ego_indices

    def clear_buffer(self):
        self.buffer = PERBuffer(
            capacity=self.config.buffer_capacity,
            min_start_train=self.config.batch_size,
            window_size=self.window_size,
            state_dim=self.TOTAL_STATE_DIM,
        )
        self.buffer.world_gen = self.world_gen

    def bump_world_gen(self, worlds):
        """resample 换掉了哪些 world，就把它们的代际 +1（其余保持不变）。"""
        for w in worlds:
            self.world_gen[int(w)] = self.world_gen.get(int(w), 0) + 1
        self.buffer.world_gen = self.world_gen

    def save(self, save_info: Dict[str, Any]) -> None:
        if self.globe_eps % self.config.save_freq == 0:
            logger.info(f'保存模型: globe_eps={self.globe_eps}, global_step={self.global_step}')
            model = {'actor': self.actor, 'critic': self.critic}
            optimizer = {'actor_optim': self.actor_optimizer, 'critic_optim': self.critic_optimizer}
            extra_info = {
                'config': self.config,
                'global_step': self.global_step,
                'history': self.history_loss,
                'globe_eps': self.globe_eps,
                'state_dim': self.TOTAL_STATE_DIM,
                'rho': self.rho,
                'gep_iteration': self.gep_iteration,
                'state_mode': self.state_mode,
                'ref_mode': self.ref_mode,
                'input_norm': self.input_norm.state_dict(),
                'actor_scheduler_last_epoch': self.actor_scheduler.last_epoch,
                'critic_scheduler_last_epoch': self.critic_scheduler.last_epoch,
            }
            metrics = {'episode': extra_info['globe_eps']}
            save_checkpoint(model=model,
                            model_name='idc-waymo-v1.0',
                            optimizer=optimizer,
                            file_dir=self.config.file_dir,
                            env_name=save_info.get('env_name', 'unknown_env'),
                            extra_info=extra_info,
                            metrics=metrics)
    

    def load(self, path: str) -> Dict[str, Any]:
        checkpoint = load_checkpoint(
            model={'actor': self.actor, 'critic': self.critic},
            filepath=path,
            optimizer={'actor_optim': self.actor_optimizer, 'critic_optim': self.critic_optimizer},
            device=self.device
        )
        loaded_dim = checkpoint.get('state_dim', self.TOTAL_STATE_DIM)
        if loaded_dim != self.TOTAL_STATE_DIM:
            logger.warning(f"加载模型维度{loaded_dim}与当前{self.TOTAL_STATE_DIM}不一致")
        self.globe_eps = checkpoint['globe_eps']
        self.history_loss = checkpoint['history']
        self.epoch_history_pdms = checkpoint.get('history_pdms', [])
        self.global_step = checkpoint['global_step']
        self.rho = checkpoint['rho'] == 0.0 and self.config.init_penalty or checkpoint['rho']
        self.gep_iteration = checkpoint['gep_iteration']
        # 归一化统计（换了表示层时维度不匹配会自动跳过）
        self.input_norm.load_state_dict(checkpoint.get('input_norm'))
        # 表示层必须跟着 checkpoint 走：用 ego 编码训的权重，评估时若按 abs 喂会静默错配
        for key in ('state_mode', 'ref_mode'):
            saved = checkpoint.get(key)
            if saved is not None and saved != getattr(self, key):
                logger.warning(f'checkpoint 里的 {key}={saved} 覆盖当前配置 '
                               f'{getattr(self, key)}')
                setattr(self, key, saved)
        if hasattr(self.state_builder, 'ref_mode'):
            self.state_builder.ref_mode = self.ref_mode
            self.state_builder.ref_window = self.ref_window

        for pg in self.actor_optimizer.param_groups:
            pg['lr'] = self.config.lr_actor_max
        for pg in self.critic_optimizer.param_groups:
            pg['lr'] = self.config.lr_critic_max

        # 重新初始化 scheduler，从 checkpoint 的 last_epoch 继续
        last_ep_a = checkpoint.get('actor_scheduler_last_epoch', 0)
        last_ep_c = checkpoint.get('critic_scheduler_last_epoch', 0)
        self.actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.actor_optimizer, T_max=self.actor_scheduler.T_max,
            eta_min=self.actor_scheduler.eta_min, last_epoch=last_ep_a)
        self.critic_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.critic_optimizer, T_max=self.critic_scheduler.T_max,
            eta_min=self.critic_scheduler.eta_min, last_epoch=last_ep_c)

        return checkpoint
        