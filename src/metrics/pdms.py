import math
import numpy as np
import torch


class PDMSScorer:
    """单世界单 episode 的 PDMS 在线累加器。

    PDMS = (NC × DAC × DDC) × (EP×5 + TTC×5 + C×2 + LK×2) / 14

    每步调用 update_step() 累加，compute() 返回最终得分。
    """

    def __init__(self, config):
        self.nc_collision_weight = getattr(config, 'pdms_nc_weight', 0)
        self.dac_off_road_threshold = getattr(config, 'pdms_dac_threshold', 3)
        self.ttc_horizon = getattr(config, 'pdms_ttc_horizon', 4.0)
        self.jerk_max = getattr(config, 'pdms_jerk_max', 10.0)

        self.w_ep = getattr(config, 'pdms_ep_weight', 5)
        self.w_ttc = getattr(config, 'pdms_ttc_weight', 5)
        self.w_comfort = getattr(config, 'pdms_comfort_weight', 2)
        self.w_lk = getattr(config, 'pdms_lk_weight', 2)
        self.weight_sum = self.w_ep + self.w_ttc + self.w_comfort + self.w_lk

        self.reset()

    def reset(self):
        self.steps = 0
        self.total_steps = 0

        self.collision_steps = 0
        self.off_road_steps = 0
        self.ddc_steps = 0

        self.ep_sum = 0.0
        self.ttc_sum = 0.0
        self.comfort_sum = 0.0
        self.lk_sum = 0.0

        self.prev_speed = None

    def update_step(self, ego_pos, ego_vel, ego_heading, partners,
                    off_road, collision, delta_phi,
                    temporal_idx, max_step,
                    road_dist_ref, lat, dt):
        """采集一个 step 的 PDMS 数据并累加。

        Args:
            ego_pos: (x, y)
            ego_vel: float, speed m/s
            ego_heading: float, rad
            partners: [N, 4] with cols (speed, rel_x, rel_y, rel_heading)
            off_road: bool
            collision: bool
            delta_phi: float, 自车 heading 与参考 heading 之差 (rad)
            temporal_idx: int, 当前步
            max_step: int, episode 总步数
            road_dist_ref: float, 参考点到最近道路边距 (≈半幅路宽)
            lat: float, 自车相对参考路径的横向偏移 (m)
            dt: float, 步长 (s)
        """
        self.steps += 1
        self.total_steps = max_step

        if collision:
            self.collision_steps += 1
        if off_road:
            self.off_road_steps += 1
        if abs(delta_phi) > math.pi / 2:
            self.ddc_steps += 1

        ep = min(1.0, temporal_idx / max(max_step, 1))

        ttc = self._compute_ttc(ego_pos, ego_vel, ego_heading, partners)
        ttc_score = min(1.0, ttc / self.ttc_horizon) if ttc < 1e9 else 1.0

        if self.prev_speed is not None and dt > 0:
            jerk = abs(ego_vel - self.prev_speed) / dt
            c_score = max(0.0, 1.0 - jerk / self.jerk_max)
        else:
            c_score = 1.0
        self.prev_speed = ego_vel

        if road_dist_ref > 1e-6:
            lk_score = max(0.0, 1.0 - abs(lat) / road_dist_ref)
        else:
            lk_score = 1.0

        self.ep_sum += ep
        self.ttc_sum += ttc_score
        self.comfort_sum += c_score
        self.lk_sum += lk_score

    def _compute_ttc(self, ego_pos, ego_vel, ego_heading, partners):
        """计算自车与最近前车的 TTC (Time to Collision)。

        返回 min TTC（秒），若无前车返回 inf。
        """
        ex, ey = ego_pos
        min_ttc = float('inf')
        cos_h = math.cos(ego_heading)
        sin_h = math.sin(ego_heading)
        for p in partners:
            p_speed, rel_x, rel_y, rel_h = float(p[0]), float(p[1]), float(p[2]), float(p[3])
            if p_speed == 0.0 and rel_x == 0.0 and rel_y == 0.0:
                continue
            px = ex + rel_x * cos_h - rel_y * sin_h
            py = ey + rel_x * sin_h + rel_y * cos_h
            dx = px - ex
            dy = py - ey
            dist = math.hypot(dx, dy)
            if dist < 1e-3:
                continue
            p_heading = ego_heading + rel_h
            p_vx = p_speed * math.cos(p_heading)
            p_vy = p_speed * math.sin(p_heading)
            unit_x = dx / dist
            unit_y = dy / dist
            closing_speed = -(ego_vel * unit_x - p_vx * unit_x + (0.0 * unit_y - p_vy * unit_y))
            closing_speed = max(0.0, closing_speed)
            if closing_speed > 1e-6:
                ttc = dist / closing_speed
                if ttc < min_ttc:
                    min_ttc = ttc
        return min_ttc

    def compute(self):
        """返回 PDMS 各维度得分字典。"""
        s = max(self.steps, 1)

        has_collision = self.collision_steps > 0
        has_off_road = self.off_road_steps > self.dac_off_road_threshold
        has_ddc = self.ddc_steps > 0

        nc = 0.0 if has_collision else 1.0
        dac = 0.0 if has_off_road else 1.0
        ddc = 0.0 if has_ddc else 1.0

        ep_avg = self.ep_sum / s
        ttc_avg = self.ttc_sum / s
        comfort_avg = self.comfort_sum / s
        lk_avg = self.lk_sum / s

        weighted = (self.w_ep * ep_avg + self.w_ttc * ttc_avg +
                    self.w_comfort * comfort_avg + self.w_lk * lk_avg) / self.weight_sum

        driving_score = nc * dac * ddc * weighted * 100.0

        return {
            'driving_score': driving_score,
            'penalties': {
                'nc': nc,
                'dac': dac,
                'ddc': ddc,
            },
            'weighted': {
                'ep': ep_avg,
                'ttc': ttc_avg,
                'comfort': comfort_avg,
                'lk': lk_avg,
            },
            'counts': {
                'collision_steps': self.collision_steps,
                'off_road_steps': self.off_road_steps,
                'ddc_steps': self.ddc_steps,
            },
            'route_completion': self.steps / max(self.total_steps, 1),
        }


class RolloutPDMSScorer:
    """评估专用：从给定 state 推演未来 H 步并计算预测 PDMS。"""

    def __init__(self, agent, config, path_idx=0):
        self.agent = agent
        self.config = config
        self.path_idx = path_idx

    def compute_rollout_pdms(self, state, world_idx, ego_idx):
        """推演 horizon 步并返回预测 PDMS。

        Returns:
            dict: predicted_pdms, plus per-step metrics list
        """
        agent = self.agent
        horizon = agent.horizon
        dt = agent.dt
        device = agent.device

        s_np = state
        s = torch.from_numpy(s_np).float().unsqueeze(0).to(device)
        w_i = [world_idx]
        p_i = [self.path_idx]
        ref_start_val = agent.DIM_EGO + agent.DIM_OTHERS + agent.DIM_VALIDITY

        scorer = PDMSScorer(self.config)

        prev_speed = float(s_np[2])

        with torch.no_grad():
            for t in range(horizon):
                u = agent.actor(s)
                s = agent.f_pred_batch(s, u, w_i, p_i)

                ego_np = s[0].cpu().numpy()
                ex, ey = float(ego_np[0]), float(ego_np[1])
                ego_vel = float(torch.hypot(s[0, 2], s[0, 3]))
                ego_heading = float(ego_np[4])

                partners = []  # 从 others 重建
                o_start = agent.DIM_EGO
                others = ego_np[o_start:o_start + agent.DIM_OTHERS].reshape(8, 4)
                for i in range(8):
                    ox, oy, oh, ov = float(others[i, 0]), float(others[i, 1]), float(others[i, 2]), float(others[i, 3])
                    if ox == 0.0 and oy == 0.0 and ov == 0.0:
                        continue
                    rel_x = ox - ex
                    rel_y = oy - ey
                    rel_h = oh - ego_heading
                    dist = math.hypot(rel_x, rel_y)
                    partners.append((dist, [ov, rel_x, rel_y, rel_h]))
                partners.sort(key=lambda p: p[0])
                partners_arr = np.array([p[1] for p in partners[:8]], dtype=np.float32)
                if partners_arr.shape[0] < 8:
                    pad = np.zeros((8 - partners_arr.shape[0], 4), dtype=np.float32)
                    partners_arr = np.vstack([partners_arr, pad]) if partners_arr.shape[0] > 0 else pad

                delta_phi = float(ego_np[ref_start_val + 1])
                lat = float(ego_np[ref_start_val + 3])
                road_dist = float(ego_np[ref_start_val + 5])
                temporal_idx = int(float(ego_np[agent.DIM_EGO + agent.DIM_OTHERS + agent.DIM_VALIDITY + agent.DIM_REF_ERROR]))

                off_road = road_dist < agent.D_road_safe
                collision = False

                scorer.update_step(
                    ego_pos=(ex, ey),
                    ego_vel=ego_vel,
                    ego_heading=ego_heading,
                    partners=partners_arr,
                    off_road=off_road,
                    collision=collision,
                    delta_phi=delta_phi,
                    temporal_idx=temporal_idx,
                    max_step=91,
                    road_dist_ref=road_dist,
                    lat=lat,
                    dt=dt,
                )
                prev_speed = ego_vel

        return scorer.compute()
