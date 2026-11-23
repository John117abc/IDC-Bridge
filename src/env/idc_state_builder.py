import numpy as np
import math
from typing import List, Dict, Tuple, Optional
import torch

from utils import get_logger
logger = get_logger('idc_state_builder')


class GPUDriveObservationBuilder:
    """
    完全基于 gpudrive 环境自带的观测接口构建 IDC 所需向量。
    依赖：
        - env.get_obs() 返回的字典（包含 road_map_obs, partner_obs 等）
        - env.sim.absolute_obs() （仅用于自车绝对位置/速度，因 ego_state 不含这些）
    """

    def __init__(self, env, scene_json_list: List[Dict]):
        """
        :param env: GPUDriveTorchEnv 实例（或任何提供 get_obs() 和 sim 的对象）
        :param scene_json_list: 保留以备目标点回退，不再用于道路点提取
        """
        self.env = env
        self.sim = env.sim
        self.num_worlds = env.num_worlds

        # 步数计数器
        self.step_counter = {w: 0 for w in range(self.num_worlds)}
        # 每 world 的"上一帧投影索引"。投影必须以它为种子；若用 step_counter 当种子，
        # max(prev, best) 会让索引永远不小于时钟，等于没改（参考点仍旧随时钟推进）。
        self._ref_idx = {w: 0 for w in range(self.num_worlds)}
        # 每 world 的 size.length（GPUDrive 拿它当轴距，逐车不同）
        self._ego_len = {w: 5.0 for w in range(self.num_worlds)}
        # 道路张量缓存（一集不变，避免 rollout 中重复 GPU→CPU 传输）
        self._road_cache = None
        # 参考点推进方式：
        #   'nearest' = 在上一进度索引附近做最近点投影（参考点随"位置"推进，默认）
        #   'index'   = 逐帧 +1（旧行为，保留做消融）
        self.ref_mode = 'nearest'
        self.ref_window = 25

        # ==========================================
        # 1. 初始化部分 (在类的 __init__ 或 reset 中)
        # ==========================================
        self.expert_pos = None
        self.expert_vel = None
        self.expert_heading = None
        self.EXPERT_TRAJ_LEN = None
        # 预处理专家轨迹数据，转换为适合快速索引的格式
        self._setup_expert_data()
        
        logger.debug(f'专家轨迹长度：{self.EXPERT_TRAJ_LEN}')

    def generate_candidate_paths(self, ego_indices, num_paths=1,
                                   num_points=91, lane_width=3.75,
                                   use_expert_path=True,
                                   trace_logs_out=None):
        """为每个 world 生成候选路径。

        use_expert_path=True:  使用 expert 轨迹（训练用）
        use_expert_path=False: 从道路图车道中心线生成（无 expert 部署用）
        trace_logs_out:        dict[world_idx -> list], 仅 use_expert_path=False 时使用
        """
        self.candidate_paths = {}
        self.num_candidate_paths = num_paths

        road_tensor = self.sim.map_observation_tensor().to_torch().cpu().numpy()

        for w in range(self.num_worlds):
            a = int(ego_indices[w])
            self.candidate_paths[w] = {}
            world_road = road_tensor[w]

            if use_expert_path:
                # ---- Expert 路径 ----
                expert_pos_w = self.expert_pos[w, a].cpu().numpy()
                expert_h_w = self.expert_heading[w, a].squeeze(-1).cpu().numpy()

                valid_len = num_points
                for j in range(1, num_points):
                    if abs(expert_pos_w[j, 0]) > 5000 or abs(expert_pos_w[j, 1]) > 5000:
                        valid_len = j
                        break
                if valid_len < num_points:
                    expert_pos_w[valid_len:, :] = expert_pos_w[valid_len - 1, :]
                    expert_h_w[valid_len:] = expert_h_w[valid_len - 1]
                    self.expert_pos[w, a] = torch.from_numpy(expert_pos_w)
                    self.expert_heading[w, a, :, :] = torch.from_numpy(expert_h_w.reshape(-1, 1))
                    logger.debug(f'[TRUNCATE] world_{w} agent_{a} expert truncated at step {valid_len}/{num_points}')

                paths = []
                path = {
                    'pos': expert_pos_w.astype(np.float32),
                    'heading': expert_h_w.astype(np.float32),
                    'speed': self._curvature_speed(expert_pos_w),
                }
                road_dists = np.zeros(num_points, dtype=np.float32)
                for j in range(num_points):
                    road_dists[j] = self._road_dist_point(
                        world_road, float(expert_pos_w[j, 0]), float(expert_pos_w[j, 1]))
                path['road_dist'] = road_dists
                paths.append(path)

            else:
                # ---- 道路图路径 ----
                ego_x = float(self.expert_pos[w, a, 0, 0].item())
                ego_y = float(self.expert_pos[w, a, 0, 1].item())
                ego_heading = float(self.expert_heading[w, a, 0, 0].item())

                tl = [] if trace_logs_out is not None else None
                path = self._generate_path_from_road_graph(
                    world_road, ego_x, ego_y, ego_heading, num_points,
                    trace_log_out=tl)
                if trace_logs_out is not None and tl is not None:
                    trace_logs_out[w] = tl
                paths = [path]

            # 指数衰减前视聚合：每个路径点的未来曲率/速度/道路预期
            decay_lam = 20
            for path in paths:
                pos_arr = path['pos']; head_arr = path['heading']
                spd_arr = path['speed']; rd_arr = path['road_dist']
                cp, sp, rp = np.zeros(91, dtype=np.float32), np.zeros(91, dtype=np.float32), np.zeros(91, dtype=np.float32)
                # 只在有效段内聚合（last_valid_idx 之后是外推段，不参与）
                nv = int(path.get('last_valid_idx', len(pos_arr) - 1))
                nv = max(1, min(nv, len(pos_arr) - 1))
                for i in range(91):
                    j_end = nv + 1
                    rem = max(j_end - i, 1)
                    weights = np.exp(-np.arange(rem) / decay_lam)
                    weights /= (weights.sum() + 1e-8)
                    if i < nv:
                        dh = head_arr[i+1:j_end] - head_arr[i:j_end-1]
                        steps = np.hypot(np.diff(pos_arr[i:j_end, 0]),
                                         np.diff(pos_arr[i:j_end, 1])) + 1e-6
                        steer_i = np.arctan(5.0 * dh / steps)
                        if len(steer_i):
                            ww = np.exp(-np.arange(len(steer_i)) / decay_lam)
                            ww /= (ww.sum() + 1e-8)
                            cp[i] = float(np.sum(ww * steer_i))
                    sp[i] = float(np.sum(weights * spd_arr[i:j_end]))
                    rp[i] = float(np.sum(weights * rd_arr[i:j_end]))
                # 有效段之后保持末值
                if nv < 90:
                    cp[nv] = cp[nv-1] if nv >= 1 else 0.0
                    cp[nv+1:] = cp[nv]
                    sp[nv+1:] = sp[nv]
                    rp[nv+1:] = rp[nv]
                path['curv_ahead'] = cp
                path['spd_ahead'] = sp
                path['road_ahead'] = rp

            self.candidate_paths[w][a] = paths

        num_worlds = self.num_worlds
        num_paths = self.num_candidate_paths
        num_points = 91

        # 构建 GPU ref_tensor 供 rollout 零 Python 循环索引
        # [num_worlds, 1, num_paths, num_points, 8]: pos_x, pos_y, speed, heading, road_dist, curv_ahead, spd_ahead, road_ahead
        ref_data = np.zeros((num_worlds, 1, num_paths, num_points, 8), dtype=np.float32)
        for w in range(num_worlds):
            a = ego_indices[w]
            for p in range(num_paths):
                path = self.candidate_paths[w][a][p]
                n = len(path['pos'])
                ref_data[w, 0, p, :n, 0] = path['pos'][:, 0]
                ref_data[w, 0, p, :n, 1] = path['pos'][:, 1]
                ref_data[w, 0, p, :n, 2] = path['speed'][:]
                ref_data[w, 0, p, :n, 3] = path['heading'][:]
                ref_data[w, 0, p, :n, 4] = path['road_dist'][:]
                ref_data[w, 0, p, :n, 5] = path['curv_ahead'][:]
                ref_data[w, 0, p, :n, 6] = path['spd_ahead'][:]
                ref_data[w, 0, p, :n, 7] = path['road_ahead'][:]
        self.ref_tensor = torch.from_numpy(ref_data).to(self.expert_pos.device)
        self._build_last_valid_lut()

        # 诊断: 打印候选路径坐标范围，确认与 ego 坐标系对齐
        for w in range(min(self.num_worlds, 3)):
            a = ego_indices[w]
            for pid, p in enumerate(self.candidate_paths[w][a]):
                pos = p['pos']
                head = p['heading']
                spd = p['speed']
                mid = len(pos) // 2
                logger.debug(f'[BEZIER-CHK] world_{w} path_{pid} '
                            f'start=({pos[0,0]:.1f},{pos[0,1]:.1f}) head={head[0]:.2f} spd={spd[0]:.2f}')
                logger.debug(f'[BEZIER-CHK] world_{w} path_{pid} '
                            f'mid=({pos[mid,0]:.1f},{pos[mid,1]:.1f}) head={head[mid]:.2f} spd={spd[mid]:.2f}')
                logger.debug(f'[BEZIER-CHK] world_{w} path_{pid} '
                            f'end=({pos[-1,0]:.1f},{pos[-1,1]:.1f}) head={head[-1]:.2f} spd={spd[-1]:.2f}')

        # 最近点跳变检测
        self._last_nearest = {}
        self._last_ego = {}

    # ========================================
    # 道路图路径生成（col6 type 3 = 车道中心线, type 1 = 道路边界）
    # ========================================

    @staticmethod
    def _group_road_segments(pts, type_col=6):
        """按 id (column 7) 分组道路点，组内按距离排序为连续折线。
        返回 {id: np.ndarray[N_i, 9]}。
        """
        ids = np.unique(pts[:, 7])
        groups = {}
        for gid in ids:
            mask = pts[:, 7] == gid
            group = pts[mask].copy()
            if len(group) > 1:
                order = [0]
                used = {0}
                while len(order) < len(group):
                    last = group[order[-1], [0, 1]]
                    best_i, best_dist = -1, float("inf")
                    for i in range(len(group)):
                        if i in used:
                            continue
                        d = math.hypot(group[i, 0] - last[0], group[i, 1] - last[1])
                        if d < best_dist:
                            best_dist, best_i = d, i
                    order.append(best_i)
                    used.add(best_i)
                group = group[np.array(order, dtype=int)]
            groups[int(gid)] = group
        return groups

    @staticmethod
    def _find_nearest_on_segments(segments, qx, qy, q_heading=None):
        """在所有车道段的所有点中找到离查询点最近的点。
        先硬过滤方向偏差 >45° 的段，再搜最近点（带边端惩罚 + 航向对齐）。
        返回 (seg_id, point_index, distance_m) 或 (None, 0, inf)。
        """
        # 阶段 1：硬过滤 —— 只保留与 ego 航向偏差 < 45° 的段
        if q_heading is not None and len(segments) > 3:
            aligned = {}
            for seg_id, pts in segments.items():
                mean_sin = np.sin(pts[:, 5]).mean()
                mean_cos = np.cos(pts[:, 5]).mean()
                mean_h = math.atan2(float(mean_sin), float(mean_cos))
                diff = abs(math.atan2(math.sin(mean_h - q_heading),
                                       math.cos(mean_h - q_heading)))
                if diff < math.radians(45):
                    aligned[seg_id] = pts
            if len(aligned) >= 3:
                segments = aligned

        best_seg, best_idx, best_score = None, 0, float("inf")
        for seg_id, pts in segments.items():
            dx = pts[:, 0] - qx
            dy = pts[:, 1] - qy
            dists = np.hypot(dx, dy)
            n = len(pts)
            for i in range(n):
                d = float(dists[i])
                # 边缘惩罚：偏向段中部，避免入口剩余点数太少
                if i == n - 1:
                    d += 15.0       # 段尾：最差
                elif i == n - 2:
                    d += 5.0        # 倒数第二：勉强
                elif i == 0:
                    d += 2.0        # 段首：可接受但非最优
                elif i == 1 and n > 3:
                    d += 0.5        # 第二点：稍弱

                # 航向对齐奖励
                if q_heading is not None:
                    seg_h = float(pts[i, 5])
                    h_diff = abs(math.atan2(math.sin(seg_h - q_heading),
                                             math.cos(seg_h - q_heading)))
                    d += h_diff * 15.0  # ~25° diff = ~6.5m penalty

                if d < best_score:
                    best_score = d
                    best_seg = seg_id
                    best_idx = i
        return best_seg, best_idx, best_score

    def _trace_lane_forward(self, segments, entry_seg_id, entry_idx,
                               max_pts=300, trace_log=None,
                               original_heading=None):
        """从入口点沿车道图向前追踪，返回 raw_trace: [(x, y, heading), ...]

        original_heading: ego 初始航向，用于全局方向漂移惩罚。
        trace_log: 传入 list 即收集每步追踪信息供诊断用；传入 None 则零开销。
        """
        raw = []
        current_seg = entry_seg_id
        current_idx = entry_idx
        recent_visited = []    # 最近 5 个段，防止短期循环
        fully_consumed = set() # 消费 ≥5 点的段永久禁止重入（防止长期循环）

        while len(raw) < max_pts:
            pts = segments[current_seg]
            seg_start_x = float(pts[current_idx, 0])
            seg_start_y = float(pts[current_idx, 1])
            seg_start_h = float(pts[current_idx, 5])

            # 从 current_idx 走到段尾
            seg_consumed = 0
            for i in range(current_idx, len(pts)):
                raw.append((float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 5])))
                seg_consumed += 1

            # 如果消费了足够多的点，标记为永久禁止重入
            if seg_consumed >= 5:
                fully_consumed.add(current_seg)
                if current_seg in recent_visited:
                    recent_visited.remove(current_seg)

            # 段尾坐标
            end_x, end_y = float(pts[-1, 0]), float(pts[-1, 1])
            end_heading = float(pts[-1, 5])

            # 段内长度（从入口到段尾）
            seg_dx = pts[current_idx:, 0] - np.roll(pts[current_idx:, 0], 1)
            seg_dy = pts[current_idx:, 1] - np.roll(pts[current_idx:, 1], 1)
            seg_dx[0] = 0.0
            seg_dy[0] = 0.0
            seg_length = float(np.sum(np.hypot(seg_dx, seg_dy)))

            # 段内总转向
            seg_turn = float(np.degrees(math.atan2(
                math.sin(end_heading - seg_start_h),
                math.cos(end_heading - seg_start_h))))

            step_info = {
                "step": len(trace_log) if trace_log is not None else 0,
                "seg_id": current_seg,
                "seg_entry_idx": current_idx,
                "seg_total_pts": len(pts),
                "seg_consumed": seg_consumed,
                "seg_length": seg_length,
                "seg_start_h_deg": float(np.degrees(seg_start_h)),
                "seg_end_h_deg": float(np.degrees(end_heading)),
                "seg_turn_deg": seg_turn,
                "end_x": end_x, "end_y": end_y,
                "end_h_deg": float(np.degrees(end_heading)),
                "cumulated_pts": len(raw),
                "next_seg": None,
                "next_idx": None,
                "next_dist": None,
                "next_dh_deg": None,
                "next_fwd_score": None,
                "candidates": [],
                "best_score": None,
                "dead_end": False,
                "extrapolated_pts": 0,
            }

            recent_visited.append(current_seg)
            if len(recent_visited) > 5:
                recent_visited.pop(0)

            # 段尾 → 找下一段（两阶段搜索）
            candidate_log = [] if trace_log is not None else None
            visited_ids = set(recent_visited) | fully_consumed

            # 阶段一：标准搜索（max 20m, dh < 60°）
            next_seg, next_idx, _ = self._find_continuation(
                segments, end_x, end_y, end_heading,
                visited_ids=visited_ids,
                candidate_log=candidate_log,
                original_heading=original_heading,
                max_dist=20.0, max_dh=math.radians(60))

            # 阶段二：宽域搜索（max 50m, dh < 30°），仅当阶段一未找到
            if next_seg is None:
                far_log = [] if trace_log is not None else None
                next_seg, next_idx, _ = self._find_continuation(
                    segments, end_x, end_y, end_heading,
                    visited_ids=visited_ids,
                    candidate_log=far_log,
                    original_heading=original_heading,
                    max_dist=50.0, max_dh=math.radians(30))
                if trace_log is not None and far_log is not None:
                    candidate_log.extend(far_log)

            if next_seg is None:
                # 死路：不外推，让重采样自然收紧到最后一个有效道路点
                step_info["dead_end"] = True
                step_info["extrapolated_pts"] = 0
                step_info["candidates"] = candidate_log or []
                if trace_log is not None:
                    trace_log.append(step_info)
                break

            # Gap 插值（段尾 → 下一段入口）
            next_start_x = float(segments[next_seg][next_idx, 0])
            next_start_y = float(segments[next_seg][next_idx, 1])
            gap_dist = math.hypot(next_start_x - end_x, next_start_y - end_y)
            gap_heading = math.atan2(next_start_y - end_y, next_start_x - end_x)

            step_info["next_seg"] = next_seg
            step_info["next_idx"] = next_idx
            step_info["next_dist"] = float(gap_dist)
            step_info["next_dh_deg"] = float(np.degrees(math.atan2(
                math.sin(float(segments[next_seg][next_idx, 5]) - end_heading),
                math.cos(float(segments[next_seg][next_idx, 5]) - end_heading))))
            step_info["candidates"] = candidate_log or []

            # 择出 best 候选的分数
            best_score = -float("inf")
            for c in (candidate_log or []):
                if not any([c.get("skipped_dist"), c.get("skipped_fwd"),
                            c.get("skipped_dh"), c.get("skipped_visited")]):
                    best_score = max(best_score, c["score"])
            step_info["best_score"] = best_score if best_score > -float("inf") else None

            if trace_log is not None:
                trace_log.append(step_info)

            n_gap = max(1, int(gap_dist / 0.5))
            for k in range(1, n_gap):
                t = k / n_gap
                raw.append((end_x + t * (next_start_x - end_x),
                             end_y + t * (next_start_y - end_y),
                             gap_heading))

            current_seg = next_seg
            current_idx = next_idx

        return raw

    @staticmethod
    def _find_continuation(segments, end_x, end_y, end_heading,
                             visited_ids=None, candidate_log=None,
                             original_heading=None,
                             max_dist=20.0, max_dh=math.radians(60)):
        """在未访问的段中搜索最佳的下一个段。
        评分加权：距离 + 方向连续性 + 前方验证 + 全局方向漂移惩罚。
        返回 (seg_id, entry_idx, score) 或 (None, 0, -inf)。

        original_heading: ego 初始航向，用于惩罚全局方向漂移（防止绕圈）。
        max_dist: 候选段入口距离上限 (m)。
        max_dh: 段入口方向与段尾方向差上限 (rad)。
        candidate_log: 传入 list 即收集每个候选信息供诊断用；传入 None 则零开销。
        """
        best = (None, 0, -float("inf"))
        cos_h = math.cos(end_heading)
        sin_h = math.sin(end_heading)

        for seg_id, pts in segments.items():
            seg_visited = visited_ids and seg_id in visited_ids
            if seg_visited:
                if candidate_log is not None:
                    candidate_log.append({
                        "seg_id": seg_id, "entry_idx": 0,
                        "dist": -1, "fwd": -1, "dh_deg": -1, "score": -1,
                        "drift_deg": -1,
                        "skipped_dist": False, "skipped_fwd": False,
                        "skipped_dh": False, "skipped_visited": True,
                    })
                continue

            for i in range(min(3, len(pts))):
                sx, sy = float(pts[i, 0]), float(pts[i, 1])
                dx, dy = sx - end_x, sy - end_y
                dist = math.hypot(dx, dy)
                skipped_dist = dist > max_dist
                if skipped_dist:
                    continue

                fwd = dx * cos_h + dy * sin_h
                skipped_fwd = fwd < -3.0
                if skipped_fwd:
                    continue

                seg_h = float(pts[i, 5])
                dh = abs(math.atan2(math.sin(seg_h - end_heading),
                                     math.cos(seg_h - end_heading)))
                skipped_dh = dh > max_dh
                if skipped_dh:
                    continue

                score = 10.0 / (dist + 0.1) - dh / math.pi + 0.5 * (fwd > 0.0)

                # 全局方向漂移惩罚：累积偏离原始方向越大，惩罚越重
                drift_deg = 0.0
                if original_heading is not None:
                    drift_deg = float(np.degrees(abs(math.atan2(
                        math.sin(seg_h - original_heading),
                        math.cos(seg_h - original_heading)))))
                    score -= drift_deg / 360.0 * 5.0

                if candidate_log is not None:
                    candidate_log.append({
                        "seg_id": seg_id, "entry_idx": i,
                        "dist": float(dist), "fwd": float(fwd),
                        "dh_deg": float(np.degrees(dh)), "score": float(score),
                        "drift_deg": drift_deg,
                        "skipped_dist": False, "skipped_fwd": False,
                        "skipped_dh": False, "skipped_visited": False,
                    })

                if score > best[2]:
                    best = (seg_id, i, score)

        return best

    @staticmethod
    def _resample_equal_arc(raw_trace, num_points):
        """等弧长重采样 raw_trace 到 num_points 个点。
        raw_trace: [(x, y, heading), ...]
        返回 pos (N,2), heading (N,)。
        """
        if len(raw_trace) < 2:
            pos = np.tile(np.array([raw_trace[0][0], raw_trace[0][1]], dtype=np.float32),
                           (num_points, 1))
            heading = np.full(num_points, raw_trace[0][2], dtype=np.float32)
            return pos, heading

        pts = np.array([[p[0], p[1]] for p in raw_trace], dtype=np.float64)
        heads = np.array([p[2] for p in raw_trace], dtype=np.float32)
        diffs = np.diff(pts, axis=0)
        dists = np.hypot(diffs[:, 0], diffs[:, 1])
        cumulative = np.zeros(len(pts), dtype=np.float64)
        cumulative[1:] = np.cumsum(dists)
        total_len = cumulative[-1]

        if total_len < 0.1:
            pos = np.tile(pts[0].astype(np.float32), (num_points, 1))
            heading = np.full(num_points, heads[0], dtype=np.float32)
            return pos, heading

        sample_arcs = np.linspace(0, total_len, num_points)
        out_pos = np.zeros((num_points, 2), dtype=np.float32)
        out_head = np.zeros(num_points, dtype=np.float32)

        seg_idx = 0
        for i, arc in enumerate(sample_arcs):
            while seg_idx < len(cumulative) - 1 and cumulative[seg_idx + 1] < arc:
                seg_idx += 1
            if seg_idx >= len(cumulative) - 1:
                out_pos[i] = pts[-1].astype(np.float32)
                out_head[i] = heads[-1]
                continue

            seg_len = cumulative[seg_idx + 1] - cumulative[seg_idx]
            t = (arc - cumulative[seg_idx]) / (seg_len + 1e-8)
            t = max(0.0, min(1.0, t))
            out_pos[i, 0] = float(pts[seg_idx, 0] + t * (pts[seg_idx + 1, 0] - pts[seg_idx, 0]))
            out_pos[i, 1] = float(pts[seg_idx, 1] + t * (pts[seg_idx + 1, 1] - pts[seg_idx, 1]))
            out_head[i] = heads[seg_idx] + t * (heads[seg_idx + 1] - heads[seg_idx])

        return out_pos, out_head

    @staticmethod
    def _min_dist_to_pts(ref_pts, qx, qy):
        """查询点到参考点集的最小距离（用于 road_dist 计算）。"""
        if len(ref_pts) == 0:
            return 10.0
        dx = ref_pts[:, 0] - qx
        dy = ref_pts[:, 1] - qy
        return float(np.sqrt((dx * dx + dy * dy).min()))

    def _generate_path_from_road_graph(self, world_road, ego_x, ego_y,
                                          ego_heading, num_points=91,
                                          trace_log_out=None):
        """从道路图车道中心线 (col6=3) 生成一条参考路径。
        返回 dict: {pos, heading, speed, road_dist}。

        trace_log_out: 传入 list 即收集追踪日志供诊断用；传入 None 则零开销。
        """
        # 1. 提取车道中心线 (col6=3) 和道路边界 (col6=1)
        lane_mask = world_road[:, 6] == 3
        edge_mask = world_road[:, 6] == 1
        lane_pts = world_road[lane_mask]
        edge_pts = world_road[edge_mask]

        if len(lane_pts) == 0:
            # 降级：直线外推
            logger.warning(f'[RG-PATH] 无车道中心线点，使用直线外推回退')
            cos_h = math.cos(ego_heading)
            sin_h = math.sin(ego_heading)
            pos = np.array([[ego_x + i * cos_h, ego_y + i * sin_h]
                             for i in range(num_points)], dtype=np.float32)
            heading = np.full(num_points, ego_heading, dtype=np.float32)
            speed = self._curvature_speed(pos)
            road_dists = np.full(num_points, 10.0, dtype=np.float32)
            if trace_log_out is not None:
                trace_log_out.append({"error": "no_lane_pts", "fallback": "linear_extrapolation"})
            return {'pos': pos, 'heading': heading, 'speed': speed, 'road_dist': road_dists}

        # 2. 按 id 分组车道段
        lane_segments = self._group_road_segments(lane_pts)
        logger.debug(f'[RG-PATH] 车道段: {len(lane_segments)}, 车道点: {len(lane_pts)}')

        # 3. 找到 ego 最近的车道点作为入口
        entry_seg_id, entry_idx, entry_dist = self._find_nearest_on_segments(
            lane_segments, ego_x, ego_y, ego_heading)

        # 入口段信息
        entry_seg_pts = lane_segments[entry_seg_id]
        entry_seg_len = float(np.sum(np.hypot(
            np.diff(entry_seg_pts[:, 0]), np.diff(entry_seg_pts[:, 1]))))
        entry_h_diff = float(np.degrees(math.atan2(
            math.sin(float(entry_seg_pts[entry_idx, 5]) - ego_heading),
            math.cos(float(entry_seg_pts[entry_idx, 5]) - ego_heading))))
        entry_residual = len(entry_seg_pts) - entry_idx

        logger.debug(f'[RG-PATH] 入口: seg={entry_seg_id} idx={entry_idx}/{len(entry_seg_pts)} '
                     f'dist={entry_dist:.1f}m seg_len={entry_seg_len:.1f}m '
                     f'h_diff={entry_h_diff:.1f}° residual={entry_residual}pts')

        # 4. 向前追踪
        trace_log = [] if trace_log_out is not None else None
        raw_trace = self._trace_lane_forward(
            lane_segments, entry_seg_id, entry_idx,
            max_pts=max(200, num_points * 2),
            trace_log=trace_log,
            original_heading=ego_heading)

        # 路径起点补回 ego 位置，让重采样自动对齐
        raw_trace.insert(0, (float(ego_x), float(ego_y), float(ego_heading)))

        # 路径过短时填充到最低可用长度（~30m）
        if len(raw_trace) >= 2:
            arc_total = sum(math.hypot(raw_trace[i][0] - raw_trace[i-1][0],
                                       raw_trace[i][1] - raw_trace[i-1][1])
                           for i in range(1, len(raw_trace)))
            if arc_total < 30.0:
                need_m = 30.0 - arc_total
                avg_sin = sum(math.sin(p[2]) for p in raw_trace) / len(raw_trace)
                avg_cos = sum(math.cos(p[2]) for p in raw_trace) / len(raw_trace)
                avg_h = math.atan2(avg_sin, avg_cos)
                last_x, last_y = raw_trace[-1][0], raw_trace[-1][1]
                for k in range(1, int(need_m) + 1):
                    raw_trace.append((last_x + k * 1.0 * math.cos(avg_h),
                                       last_y + k * 1.0 * math.sin(avg_h),
                                       avg_h))
                logger.debug(f'[RG-PATH] 路径过短 ({arc_total:.0f}m)，填充至 {30.0:.0f}m')

        # 5. 等弧长重采样
        pos, heading = self._resample_equal_arc(raw_trace, num_points)

        # 6. 曲率限速
        speed = self._curvature_speed(pos)

        # 7. 道路距离
        road_dists = np.zeros(num_points, dtype=np.float32)
        for j in range(num_points):
            road_dists[j] = self._min_dist_to_pts(edge_pts, float(pos[j, 0]), float(pos[j, 1]))

        # 8. 导出追踪日志
        if trace_log_out is not None and trace_log is not None:
            real_pts = sum(1 for si in trace_log for _ in [] if True)  # placeholder — computed below
            real_pts = sum(si["seg_consumed"] for si in trace_log
                           if not si.get("dead_end"))
            extrap_pts = (trace_log[-1].get("extrapolated_pts", 0)
                          if trace_log and trace_log[-1].get("dead_end") else 0)
            max_gap = 0.0
            for j in range(1, num_points):
                g = float(np.hypot(pos[j, 0] - pos[j-1, 0],
                                   pos[j, 1] - pos[j-1, 1]))
                max_gap = max(max_gap, g)

            trace_log_out.append({
                "world_meta": {
                    "ego_x": float(ego_x), "ego_y": float(ego_y),
                    "ego_h_deg": float(np.degrees(ego_heading)),
                    "lane_pts": len(lane_pts), "edge_pts": len(edge_pts),
                    "segments": len(lane_segments),
                },
                "entry": {
                    "seg_id": entry_seg_id, "entry_idx": entry_idx,
                    "seg_total_pts": len(entry_seg_pts),
                    "seg_length": entry_seg_len,
                    "dist_to_ego": float(entry_dist),
                    "h_diff_deg": entry_h_diff,
                    "residual_pts": entry_residual,
                },
                "trace": trace_log,
                "quality": {
                    "raw_total_pts": len(raw_trace),
                    "real_pts": real_pts,
                    "extrapolated_pts": extrap_pts,
                    "extrapolated_pct": extrap_pts / len(raw_trace) * 100 if len(raw_trace) > 0 else 0,
                    "path_start": (float(pos[0, 0]), float(pos[0, 1])),
                    "path_end": (float(pos[-1, 0]), float(pos[-1, 1])),
                    "path_length": float(np.sum(np.hypot(
                        np.diff(pos[:, 0]), np.diff(pos[:, 1])))),
                    "max_gap": max_gap,
                    "path_h_start_deg": float(np.degrees(heading[0])),
                    "path_h_end_deg": float(np.degrees(heading[-1])),
                },
            })

        return {
            'pos': pos,
            'heading': heading,
            'speed': speed,
            'road_dist': road_dists,
        }

    # ========================================
    # 曲率限速（纯静态，无专家决策信息）
    # ========================================

    @staticmethod
    def _curvature_speed(pos: np.ndarray, v_max: float = 20.0,
                          a_lat_max: float = 2.5) -> np.ndarray:
        """从路径位置的数值曲率计算曲率限速 [num_points]"""
        n = len(pos)
        speeds = np.full(n, v_max, dtype=np.float32)
        for i in range(1, n - 1):
            dx1 = pos[i, 0] - pos[i-1, 0]; dy1 = pos[i, 1] - pos[i-1, 1]
            dx2 = pos[i+1, 0] - pos[i, 0]; dy2 = pos[i+1, 1] - pos[i, 1]
            ds1 = float(np.hypot(dx1, dy1)) + 1e-6
            ds2 = float(np.hypot(dx2, dy2)) + 1e-6
            kappa = abs(np.arctan2(dy2, dx2) - np.arctan2(dy1, dx1)) / ((ds1 + ds2) / 2.0)
            speeds[i] = float(np.sqrt(a_lat_max / (kappa + 1e-6)))
        speeds = np.minimum(speeds, v_max)
        speeds[0] = speeds[1]
        speeds[-1] = speeds[-2]
        return speeds

    def rebuild_paths_tensor(self):
        """由 self.candidate_paths 重算"前视聚合"与 ref_tensor。

        为什么不改 generate_candidate_paths：
        它内部把这两步写死在世界循环里，直接改会动到基线路径。这里复制一份
        独立方法，只给"外部注入候选"（candidate_sampler.install_candidates）
        和以后的多候选训练用，基线行为完全不受影响。
        """
        decay_lam = 20
        for w in range(self.num_worlds):
            for a, paths in self.candidate_paths.get(w, {}).items():
                for path in paths:
                    pos_arr = path['pos']
                    head_arr = path['heading']
                    spd_arr = path['speed']
                    rd_arr = path['road_dist']
                    n = len(pos_arr)
                    # 有效长度：last_valid_idx 之后是外推段，不参与聚合
                    n = min(n, int(path.get('last_valid_idx', n - 1)) + 1)
                    cp = np.zeros(91, dtype=np.float32)
                    sp = np.zeros(91, dtype=np.float32)
                    rp = np.zeros(91, dtype=np.float32)
                    for i in range(91):
                        rem = min(91, n) - i
                        if rem <= 0:
                            cp[i] = cp[i - 1]
                            sp[i] = sp[i - 1]
                            rp[i] = rp[i - 1]
                            continue
                        weights = np.exp(-np.arange(rem) / decay_lam)
                        weights /= (weights.sum() + 1e-8)
                        if i < n - 1:
                            dh = head_arr[i + 1:min(n, 91)] - head_arr[i:min(n, 91) - 1]
                            steps = np.hypot(np.diff(pos_arr[i:min(n, 91), 0]),
                                             np.diff(pos_arr[i:min(n, 91), 1])) + 1e-6
                            steer_i = np.zeros(len(dh), dtype=np.float32)
                            for k in range(min(len(dh), len(weights))):
                                steer_i[k] = float(np.arctan(5.0 * dh[k]
                                                             / max(steps[k], 1e-6)))
                            cp[i] = float(np.sum(weights[:len(steer_i)] * steer_i))
                        sp[i] = float(np.sum(weights * spd_arr[i:i + rem]))
                        rp[i] = float(np.sum(weights * rd_arr[i:i + rem]))
                    path['curv_ahead'] = cp
                    path['spd_ahead'] = sp
                    path['road_ahead'] = rp

        num_paths = self.num_candidate_paths
        num_points = 91
        ref_data = np.zeros((self.num_worlds, 1, num_paths, num_points, 8),
                            dtype=np.float32)
        for w in range(self.num_worlds):
            for a, paths in self.candidate_paths.get(w, {}).items():
                for p in range(min(num_paths, len(paths))):
                    path = paths[p]
                    n = min(len(path['pos']), num_points)
                    ref_data[w, 0, p, :n, 0] = path['pos'][:n, 0]
                    ref_data[w, 0, p, :n, 1] = path['pos'][:n, 1]
                    ref_data[w, 0, p, :n, 2] = path['speed'][:n]
                    ref_data[w, 0, p, :n, 3] = path['heading'][:n]
                    ref_data[w, 0, p, :n, 4] = path['road_dist'][:n]
                    ref_data[w, 0, p, :n, 5] = path['curv_ahead'][:n]
                    ref_data[w, 0, p, :n, 6] = path['spd_ahead'][:n]
                    ref_data[w, 0, p, :n, 7] = path['road_ahead'][:n]
        self.ref_tensor = torch.from_numpy(ref_data).to(self.expert_pos.device)
        self._build_last_valid_lut()
        logger.debug(f'[REBUILD] ref_tensor 重建: worlds={self.num_worlds} '
                     f'paths={num_paths}')

    def _setup_expert_data(self):
        # 假设 sim.expert_trajectory_tensor() 返回形状为 [num_worlds, num_agents, 16*91]
        raw_traj = self.sim.expert_trajectory_tensor().to_torch()
        W, A, total_feats = raw_traj.shape
        T = 91  # 根据官方代码确认为 91
        self.EXPERT_TRAJ_LEN = T

        # 按照官方定义的 Offset 一次性切片 (利用 GPU 向量化操作)
        # 形状都会变为 [W, A, T, dim]
        self.expert_pos = raw_traj[:, :, :2*T].reshape(W, A, T, 2)
        self.expert_vel = raw_traj[:, :, 2*T:4*T].reshape(W, A, T, 2)
        self.expert_heading = raw_traj[:, :, 4*T:5*T].reshape(W, A, T, 1)

    def get_expert_steer_batch(self, world_indices, temporal_indices, ego_indices_map):
        """从 expert pos/heading 数值差分计算近似 steering [batch]"""
        steers = []
        for i in range(len(world_indices)):
            w = world_indices[i]
            a = ego_indices_map[w]
            t = min(int(temporal_indices[i]), self.EXPERT_TRAJ_LEN - 2)
            dh = float(self.expert_heading[w, a, t + 1] - self.expert_heading[w, a, t])
            dx = float(self.expert_pos[w, a, t + 1, 0] - self.expert_pos[w, a, t, 0])
            dy = float(self.expert_pos[w, a, t + 1, 1] - self.expert_pos[w, a, t, 1])
            ds = math.hypot(dx, dy) + 1e-6
            # kinematic bicycle model: δ = atan(L × Δθ / Δs)
            # L 用该车自己的 size.length（GPUDrive 拿车长当轴距，逐车不同）
            L = float(self._ego_len.get(w, 5.0))
            steers.append(math.atan(L * dh / ds))
        return torch.tensor(steers, dtype=torch.float32)

    def clear_cache(self):
        """每集开始时调用，使道路缓存失效。"""
        self._road_cache = None

    def reset_world_step(self, world_idx: int, step: int = 0):
        self.step_counter[world_idx] = step
        self._ref_idx[world_idx] = step

    def increment_step(self, world_idx: int):
        self.step_counter[world_idx] += 1

    # ====================== 主入口 ======================
    def get_idc_observation(self, 
                            world_idx: int, 
                            agent_idx: int,
                            num_other_vehicles: int = 8,
                            path_idx: int = 0) -> np.ndarray:
        s_ego = self.get_ego_state(world_idx, agent_idx)
        s_others = self.get_other_vehicles(world_idx, agent_idx, num_other_vehicles)
        s_ref = self.get_ref_state_from_path(world_idx, agent_idx, path_idx,
                                              ego_x=s_ego[0], ego_y=s_ego[1])
        s_ref_error = self._calc_ref_error(s_ego, s_ref)

        network_state = np.concatenate([
            s_ego,
            s_others.flatten().astype(np.float32),
            s_ref_error.astype(np.float32)
        ])
        return network_state

    def _get_idc_obs_v1(self, ego_indices: list,
                       num_other_vehicles: int = 8,
                       path_indices: list = None,
                       _abs_np=None, _rel_np=None,
                       _partner_np=None) -> list:
        """
        批量构建所有 world 的网络状态。
        path_indices: 可选，list of int，每个 world 使用的候选路径索引。
        _abs_np/_rel_np/_partner_np: 可选，预拉取的 tensor（避免重复 GPU→CPU 传输）
        """
        if _abs_np is None:
            abs_np = self.sim.absolute_self_observation_tensor().to_torch().cpu().numpy()
        else:
            abs_np = _abs_np
        if _rel_np is None:
            rel_np = self.sim.self_observation_tensor().to_torch().cpu().numpy()
        else:
            rel_np = _rel_np
        if _partner_np is None:
            partner_np = self.sim.partner_observations_tensor().to_torch().cpu().numpy()
        else:
            partner_np = _partner_np

        states = []
        for w in range(self.num_worlds):
            aidx = ego_indices[w]
            x = float(abs_np[w, aidx, 0])
            y = float(abs_np[w, aidx, 1])
            heading = float(abs_np[w, aidx, 7])
            speed = float(rel_np[w, aidx, 0])
            if not (np.isfinite(x) and np.isfinite(y)
                    and np.isfinite(heading) and np.isfinite(speed)):
                logger.warning(f'world_{w} 自车观测含 inf/nan，整条置零')
                x = y = heading = speed = 0.0
            ego = np.array([x, y, speed, 0.0, heading, 0.0], dtype=np.float32)
            # GPUDrive 用 size.length 当轴距，实测就等于 abs_obs[...,10]
            veh_len = float(abs_np[w, aidx, 10])
            if not np.isfinite(veh_len) or not (1.0 < veh_len < 15.0):
                veh_len = 5.286          # 实测中位数兜底
            self._ego_len[w] = veh_len

            partners = partner_np[w, aidx]
            cos_h = np.cos(heading)
            sin_h = np.sin(heading)
            others_list = []
            for i in range(partners.shape[0]):
                p = partners[i]
                p_speed = p[0]
                rel_x = p[1]
                rel_y = p[2]
                rel_h = p[3]
                # 跳过 GPUDrive 空槽位: speed=0, rel_x=0, rel_y=0 的 phantom 车
                if p_speed == 0.0 and rel_x == 0.0 and rel_y == 0.0:
                    continue
                # 空槽位/损坏场景可能带 inf 或哨兵值；这类样本会让整条 rollout 变 NaN
                if not (np.isfinite(p_speed) and np.isfinite(rel_x)
                        and np.isfinite(rel_y) and np.isfinite(rel_h)):
                    continue
                if abs(rel_x) > 1e3 or abs(rel_y) > 1e3 or abs(p_speed) > 1e2:
                    continue
                abs_h = heading + rel_h
                gx = x + rel_x * cos_h - rel_y * sin_h
                gy = y + rel_x * sin_h + rel_y * cos_h
                dist = np.hypot(gx - x, gy - y)
                others_list.append((dist, [gx, gy, abs_h, p_speed]))
            others_list.sort(key=lambda t: t[0])
            others = np.array([d[1] for d in others_list[:num_other_vehicles]], dtype=np.float32)
            # 【修正 S1】真实车数量必须在补零之前统计。原实现先 vstack 补成
            # num_other_vehicles 行、再算 n_real，导致 n_real 恒等于 8、validity 全 1，
            # 于是 (0,0) 上的零填充"幽灵车"被当成真车进入双圆碰撞约束。
            n_real = int(others.shape[0]) if others.ndim == 2 else 0
            validity = np.zeros(num_other_vehicles, dtype=np.float32)
            validity[:n_real] = 1.0
            if n_real == 0:
                others = np.zeros((num_other_vehicles, 4), dtype=np.float32)
            elif n_real < num_other_vehicles:
                others = np.vstack([others, np.zeros((num_other_vehicles - n_real, 4),
                                                     dtype=np.float32)])

            pid = path_indices[w] if path_indices is not None else 0
            # 进度索引：'nearest' 模式在上一索引窗口内做最近点投影（参考点随位置推进），
            #             'index' 模式直接用步数计数器（旧行为，做消融用）
            path = self.candidate_paths[w][aidx][pid]
            t_raw = max(0, min(self.step_counter[w], len(path['pos']) - 1))
            if getattr(self, 'ref_mode', 'nearest') == 'nearest':
                # 以"上一帧投影索引"为种子做窗口内最近点投影（不以时钟为种子）
                prev_idx = self._ref_idx.get(w, 0)
                t = self._project_index_np(path['pos'], x, y, prev_idx,
                                           getattr(self, 'ref_window', 25))
                self._ref_idx[w] = t
            else:
                t = t_raw
                self._ref_idx[w] = t_raw
            rx, ry = float(path['pos'][t, 0]), float(path['pos'][t, 1])
            rh = float(path['heading'][t])
            # 参考速度取 t+5（前瞻 0.5 s）：路径起点的期望速度常≈0，直接取 t 会让车"起步即死锁"
            _ts = int(min(t + 5, len(path['speed']) - 1))
            rs = float(path['speed'][_ts])
            ref = np.array([rx, ry, rs, 0.0, rh, 0.0], dtype=np.float32)
            ref_err = self._calc_ref_error(ego, ref)

            # 前瞻参考点（t+5, t+10, t+15），给 Actor 前方弯道/道路信息
            num_pts = len(path['pos'])
            nv_la = int(path.get('last_valid_idx', num_pts - 1))
            nv_la = max(1, min(nv_la, num_pts - 1))
            t_l1 = min(t + 5, nv_la)
            t_l2 = min(t + 10, nv_la)
            t_l3 = min(t + 15, nv_la)

            lx1, ly1 = float(path['pos'][t_l1, 0]), float(path['pos'][t_l1, 1])
            lh1 = float(path['heading'][t_l1])
            lat1 = (ly1 - y) * math.cos(lh1) - (lx1 - x) * math.sin(lh1)
            dphi_l1 = lh1 - ego[4]
            dphi_l1 = math.atan2(math.sin(dphi_l1), math.cos(dphi_l1))
            road1 = path['road_dist'][t_l1]
            spd1 = path['speed'][t_l1]

            lx2, ly2 = float(path['pos'][t_l2, 0]), float(path['pos'][t_l2, 1])
            lh2 = float(path['heading'][t_l2])
            lat2 = (ly2 - y) * math.cos(lh2) - (lx2 - x) * math.sin(lh2)
            dphi_l2 = lh2 - ego[4]
            dphi_l2 = math.atan2(math.sin(dphi_l2), math.cos(dphi_l2))
            road2 = path['road_dist'][t_l2]
            spd2 = path['speed'][t_l2]

            lx3, ly3 = float(path['pos'][t_l3, 0]), float(path['pos'][t_l3, 1])
            lh3_ = float(path['heading'][t_l3])
            lat3 = (ly3 - y) * math.cos(lh3_) - (lx3 - x) * math.sin(lh3_)
            dphi_l3 = lh3_ - ego[4]
            dphi_l3 = math.atan2(math.sin(dphi_l3), math.cos(dphi_l3))
            road3 = path['road_dist'][t_l3]
            spd3 = path['speed'][t_l3]

            lookahead_err = np.array([lat1, dphi_l1, road1, spd1,
                                       lat2, dphi_l2, road2, spd2,
                                       lat3, dphi_l3, road3, spd3,
                                       path['curv_ahead'][t], path['spd_ahead'][t], path['road_ahead'][t]], dtype=np.float32)
            ref_err = np.concatenate([ref_err, lookahead_err])

            # 诊断大偏离: pos_err > 100m 时打印 ego/ref 坐标和路径索引
            if abs(ref_err[0]) > 100.0:
                logger.debug(f'[LARGE-ERR] world_{w} step={self.step_counter[w]} pos_err={ref_err[0]:.1f}m '
                               f'ego=({x:.1f},{y:.1f}) ref=({rx:.1f},{ry:.1f}) path={pid}')

            state = np.concatenate([
                ego,
                others.flatten().astype(np.float32),
                validity.astype(np.float32),
                ref_err.astype(np.float32),
                np.array([float(t)], dtype=np.float32),
                np.array([veh_len], dtype=np.float32),
            ])
            states.append(state)
        return states

    def get_ego_positions_batch(self, ego_indices: list,
                                 _abs_np=None) -> np.ndarray:
        """仅拉取 self 绝对观测，返回 [num_worlds, 2] 的 (x, y) 坐标。"""
        if _abs_np is None:
            abs_np = self.sim.absolute_self_observation_tensor().to_torch().cpu().numpy()
        else:
            abs_np = _abs_np
        pos = np.zeros((self.num_worlds, 2), dtype=np.float32)
        for w in range(self.num_worlds):
            a = ego_indices[w]
            pos[w, 0] = float(abs_np[w, a, 0])
            pos[w, 1] = float(abs_np[w, a, 1])
        return pos

    # --------------------------------------------------------
    #  自车状态 [x, y, v_lon, v_lat, heading, omega=0] (车体坐标系)
    #  注意：ego_state 不提供位置和速度分量，只能用 absolute_obs
    # --------------------------------------------------------
    def get_ego_state(self, world_idx: int, agent_idx: int) -> np.ndarray:
        # 绝对观测：位置 + 航向
        abs_tensor = self.sim.absolute_self_observation_tensor().to_torch()
        state_abs = abs_tensor[world_idx, agent_idx].cpu().numpy()
        x = state_abs[0]
        y = state_abs[1]
        heading = state_abs[7]
        
        # 相对观测：速度（标量）
        rel_tensor = self.sim.self_observation_tensor().to_torch()
        speed = rel_tensor[world_idx, agent_idx, 0].item()
        
        omega = 0.0  # 角速度不直接提供

        return np.array([x, y, speed, 0.0, heading, omega], dtype=np.float32)



    def get_other_vehicles(self, 
                            world_idx: int, 
                            ego_agent_idx: int, 
                            max_partners: int = None
                            ) -> np.ndarray:
        """
        获取周车状态（全局绝对坐标），按距离自车由近到远排序。
        
        Args:
            world_idx: 世界索引
            ego_agent_idx: 自车在场景中的索引（通常是0或可控智能体索引）
            max_partners: 最多返回多少个周车，默认None表示返回全部
        
        Returns:
            np.ndarray: 形状 (num_partners, 6)，每行 [x, y, vx, vy, heading, omega]
                        按距离自车升序排列
        """
        # 1. 获取自车状态（用于坐标转换和距离计算）
        ego_state = self.get_ego_state(world_idx, ego_agent_idx)  # [x, y, vx, vy, heading, omega]
        ego_x, ego_y = ego_state[0], ego_state[1]
        ego_heading = ego_state[4]
        
        # 2. 获取周车观测张量 (num_worlds, max_agents, max_agents-1, 9)
        partner_tensor = self.sim.partner_observations_tensor().to_torch()
        # 取出当前世界和当前自车对应的周车数据
        # 形状: (max_agents-1, 9)
        partners = partner_tensor[world_idx, ego_agent_idx].cpu().numpy()
        
        num_partners = partners.shape[0]  # = max_agents - 1
        if num_partners == 0:
            return np.empty((0, 6), dtype=np.float32)
        
        # 3. 解析每个周车
        partner_states = []
        for i in range(num_partners):
            p = partners[i]
            # 根据 PartnerObs 的解析顺序（observation.py）:
            # 0:speed, 1:rel_pos_x, 2:rel_pos_y, 3:orientation,
            # 4:vehicle_length, 5:vehicle_width, 6:vehicle_height,
            # 7:agent_type, 8:ids
            speed = p[0]
            rel_x = p[1]
            rel_y = p[2]
            rel_heading = p[3]   # 相对朝向角（弧度）
            
            # 绝对朝向角
            abs_heading = ego_heading + rel_heading
            
            # 将相对坐标转换为全局绝对坐标
            # 旋转矩阵: 自车坐标系 -> 全局坐标系
            cos_h = np.cos(ego_heading)
            sin_h = np.sin(ego_heading)
            global_x = ego_x + rel_x * cos_h - rel_y * sin_h
            global_y = ego_y + rel_x * sin_h + rel_y * cos_h
            
            # 计算速度分量
            vx = speed * np.cos(abs_heading)
            vy = speed * np.sin(abs_heading)
            
            # 距离（用于排序）
            distance = np.hypot(global_x - ego_x, global_y - ego_y)
            
            partner_states.append((distance, [global_x, global_y, abs_heading, speed]))
        
        # 4. 按距离排序
        partner_states.sort(key=lambda x: x[0])
        
        # 5. 截取前 max_partners 个
        if max_partners is not None:
            partner_states = partner_states[:max_partners]
        
        # 6. 转换为 numpy 数组
        result = np.array([state for _, state in partner_states], dtype=np.float32)  # shape (N,4)
        return result

    def get_road_edges(self, world_idx: int, ego_x: float, ego_y: float) -> np.ndarray:
        """
        返回离自车最近的 RoadLine 点（全局绝对坐标）。
        如果没有任何 RoadLine 点，返回自车坐标。
        返回格式: [x, y, 0, 0, 0, 0]
        """
        # 获取全局道路观测张量 (num_worlds, num_road_points, 9)
        road_tensor = self.sim.map_observation_tensor().to_torch()
        # 取出指定世界的所有道路点 (num_road_points, 9)
        road_points = road_tensor[world_idx].cpu().numpy()
        
        # 筛选类型为 RoadLine 的点 (type 索引为 6，值为 1)
        TYPE_IDX = 6
        ROADLINE_TYPE = 1
        line_points = road_points[road_points[:, TYPE_IDX] == ROADLINE_TYPE]
        
        if len(line_points) == 0:
            # 没有 RoadLine 点，返回自车坐标
            return np.array([ego_x, ego_y, 0., 0., 0., 0.], dtype=np.float32)
        
        # 计算每个点到自车的欧氏距离
        dx = line_points[:, 0] - ego_x
        dy = line_points[:, 1] - ego_y
        distances = np.hypot(dx, dy)
        closest_idx = np.argmin(distances)
        closest_x = line_points[closest_idx, 0]
        closest_y = line_points[closest_idx, 1]
        
        return np.array([closest_x, closest_y, 0., 0., 0., 0.], dtype=np.float32)
    

    def get_nearest_ref_point(self, world_idx: int, agent_idx: int, ego_x: float, ego_y: float):
        """
        返回专家轨迹上离自车最近点的索引、位置、航向、速度。
        """
        pos_traj = self.expert_pos[world_idx, agent_idx]  # (T, 2)
        # 计算所有点到自车的欧氏距离平方
        dx = pos_traj[:, 0] - ego_x
        dy = pos_traj[:, 1] - ego_y
        dist_sq = dx*dx + dy*dy
        nearest_idx = torch.argmin(dist_sq).item()
        
        ref_x = pos_traj[nearest_idx, 0].item()
        ref_y = pos_traj[nearest_idx, 1].item()
        ref_heading = self.expert_heading[world_idx, agent_idx, nearest_idx].item()
        ref_speed = torch.norm(self.expert_vel[world_idx, agent_idx, nearest_idx]).item()  # 速度大小
        
        return nearest_idx, (ref_x, ref_y, ref_heading, ref_speed)

    @staticmethod
    def _road_dist_point(road_tensor_world, qx, qy):
        """查询点到最近道路边界点的距离（纯函数，可复用）"""
        mask = road_tensor_world[:, 6] == 1  # TYPE_IDX=6, ROADLINE_TYPE=1
        line_pts = road_tensor_world[mask]
        if len(line_pts) == 0:
            return 10.0  # 无 RoadLine 数据时的合理 fallback（~典型道路半宽）
        dx = line_pts[:, 0] - qx
        dy = line_pts[:, 1] - qy
        nearest = int(np.argmin(dx * dx + dy * dy))
        return float(math.hypot(dx[nearest], dy[nearest]))

    def get_idc_observations_batch(self, ego_indices: list,
                                    num_other_vehicles: int = 8,
                                    path_indices: list = None,
                                    _abs_np=None, _rel_np=None,
                                    _partner_np=None) -> list:
        import os as _os
        if int(_os.environ.get('IDC_STATE_V2', '0')) >= 1:
            return self.get_idc_observations_v2(
                ego_indices, num_other_vehicles=num_other_vehicles,
                path_indices=path_indices, _abs_np=_abs_np,
                _rel_np=_rel_np, _partner_np=_partner_np)
        return self._get_idc_obs_v1(
            ego_indices, num_other_vehicles=num_other_vehicles,
            path_indices=path_indices, _abs_np=_abs_np,
            _rel_np=_rel_np, _partner_np=_partner_np)

    def get_idc_observations_v2(self, ego_indices, num_other_vehicles=8,
                                path_indices=None, _abs_np=None,
                                _rel_np=None, _partner_np=None):
        """v2 = v1 核心（66，绝对坐标）+ 追加特征（19）= 85 维。"""
        import os
        import sys
        import numpy as _np
        _tp = os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))), 'tools')
        if _tp not in sys.path:
            sys.path.insert(0, _tp)
        import idc_state_v2 as V2

        core = self._get_idc_obs_v1(
            ego_indices, num_other_vehicles=num_other_vehicles,
            path_indices=path_indices, _abs_np=_abs_np,
            _rel_np=_rel_np, _partner_np=_partner_np)

        abs_np = (_abs_np if _abs_np is not None else
                  self.sim.absolute_self_observation_tensor().to_torch().cpu().numpy())
        rel_np = (_rel_np if _rel_np is not None else
                  self.sim.self_observation_tensor().to_torch().cpu().numpy())
        partner_np = (_partner_np if _partner_np is not None else
                      self.sim.partner_observations_tensor().to_torch().cpu().numpy())

        out = []
        for w, st in enumerate(core):
            a = int(ego_indices[w])
            x = float(abs_np[w, a, 0]); y = float(abs_np[w, a, 1])
            h = float(abs_np[w, a, 7]); v = float(rel_np[w, a, 0])
            c, s_ = _np.cos(h), _np.sin(h)
            nbrs = []
            for row in partner_np[w, a]:
                pv, rx, ry, rh = (float(row[0]), float(row[1]),
                                  float(row[2]), float(row[3]))
                if pv == 0.0 and rx == 0.0 and ry == 0.0:
                    continue
                if not _np.isfinite([pv, rx, ry, rh]).all():
                    continue
                nbrs.append(dict(x=x + rx * c - ry * s_, y=y + rx * s_ + ry * c,
                                 h=h + rh, v=pv))
            nbrs.sort(key=lambda q: (q['x'] - x) ** 2 + (q['y'] - y) ** 2)
            out.append(V2.build_v2(st, (x, y, h), nbrs,
                                   step=self.step_counter[w], v_ego=v))
        return out

    def get_road_dist_batch(self, world_indices, temporal_indices,
                             ego_indices_map, path_indices, device) -> torch.Tensor:
        """批量查询预计算的道路边界距离 [batch]"""
        if hasattr(self, 'ref_tensor'):
            w_tensor = torch.tensor(list(world_indices), device=self.ref_tensor.device, dtype=torch.long)
            p_tensor = torch.tensor(list(path_indices), device=self.ref_tensor.device, dtype=torch.long)
            t_tensor = temporal_indices.long().clamp(0, self.ref_tensor.shape[3] - 1).to(self.ref_tensor.device)
            if getattr(self, 'last_valid_lut', None) is not None:
                _lv = self.last_valid_lut[w_tensor, 0, p_tensor].to(t_tensor.device)
                t_tensor = torch.minimum(t_tensor, _lv)
            return self.ref_tensor[w_tensor, 0, p_tensor, t_tensor, 4].to(device)
        # CPU 回退
        dists = []
        for i in range(len(world_indices)):
            w = world_indices[i]
            a = ego_indices_map[w]
            pid = path_indices[i]
            t = int(temporal_indices[i])
            path = self.candidate_paths[w][a][pid]
            t = max(0, min(t, len(path['road_dist']) - 1))
            dists.append(float(path['road_dist'][t]))
        return torch.tensor(dists, device=device, dtype=torch.float32)

    @staticmethod
    def _calc_ref_error(ego_state: np.ndarray, ref_state: np.ndarray) -> np.ndarray:
        dx = ref_state[0] - ego_state[0]
        dy = ref_state[1] - ego_state[1]
        delta_p = math.hypot(dx, dy)
        cross = dy * math.cos(ref_state[4]) - dx * math.sin(ref_state[4])
        delta_p *= math.copysign(1.0, cross)

        delta_phi = ref_state[4] - ego_state[4]
        delta_phi = math.atan2(math.sin(delta_phi), math.cos(delta_phi))

        delta_v = math.hypot(ego_state[2], ego_state[3]) - ref_state[2]
        return np.array([delta_p, delta_phi, delta_v], dtype=np.float32)

    # ========================================
    # 参考点投影（按位置推进，而不是按时间推进）
    # ========================================

    @staticmethod
    def _project_index_np(pos: np.ndarray, x: float, y: float,
                          prev_idx: int, window: int) -> int:
        """在 pos[prev-window : prev+window] 内找最近点，且不倒退。"""
        n = len(pos)
        prev = int(max(0, min(int(prev_idx), n - 1)))
        lo = max(0, prev - int(window))
        hi = min(n - 1, prev + int(window))
        seg = pos[lo:hi + 1]
        d = (seg[:, 0] - x) ** 2 + (seg[:, 1] - y) ** 2
        best = lo + int(np.argmin(d))
        return max(prev, best)

    def _project_index_torch(self, w_t, p_t, x, y, prev_idx, window):
        """GPU 版：在 ref_tensor 上做窗口内最近点搜索，返回 [B] long（不倒退）。"""
        rt = self.ref_tensor
        T = rt.shape[3]
        idx = torch.arange(T, device=rt.device).view(1, T)
        prev = prev_idx.long().clamp(0, T - 1)
        lo = (prev - int(window)).clamp(0, T - 1).view(-1, 1)
        hi = (prev + int(window)).clamp(0, T - 1).view(-1, 1)
        mask = (idx >= lo) & (idx <= hi)
        pos = rt[w_t, 0, p_t]                      # [B, T, 8]
        dx = pos[..., 0] - x.view(-1, 1)
        dy = pos[..., 1] - y.view(-1, 1)
        d2 = dx * dx + dy * dy
        d2 = d2.masked_fill(~mask, float('inf'))
        best = d2.argmin(dim=1)
        return torch.maximum(best, prev)


    def _build_last_valid_lut(self):
        """构建 (W,1,P) 的有效索引表，供批量读取裁剪使用。"""
        W = int(self.num_worlds)
        P = int(getattr(self, 'num_candidate_paths', 1))
        N = int(getattr(self, 'EXPERT_TRAJ_LEN', 91))
        lut = np.full((W, 1, P), N - 1, np.int64)
        for w in range(W):
            for a, paths in self.candidate_paths.get(w, {}).items():
                for p in range(min(P, len(paths))):
                    lut[w, 0, p] = int(paths[p].get('last_valid_idx', N - 1))
        self.last_valid_lut = torch.from_numpy(lut).to(self.expert_pos.device)

    def get_ref_states_batch(self, world_indices, ego_xs, ego_ys,
                               ego_indices_map: list,
                               path_indices=None,
                               temporal_indices=None,
                               ref_mode: str = 'index',
                               window: int = 25) -> torch.Tensor:
        """
        批量获取参考状态，返回 [batch, 7] tensor。
        列：[rx, ry, rs, 0, rh, 0, ref_idx]
        - temporal_indices: 当前进度索引（nearest 模式下作为搜索窗口中心）
        - ref_mode='nearest': 在窗口内做最近点投影 -> 参考点随位置推进
        - ref_mode='index'  : 直接按索引取（旧行为）
        """
        if temporal_indices is not None and path_indices is not None and hasattr(self, 'ref_tensor'):
            # GPU 快速路径：tensor 索引替代 Python 循环
            if not torch.is_tensor(temporal_indices):
                temporal_indices = torch.as_tensor(temporal_indices)
            w_tensor = torch.tensor(list(world_indices), device=self.ref_tensor.device, dtype=torch.long)
            p_tensor = torch.tensor(list(path_indices), device=self.ref_tensor.device, dtype=torch.long)
            t_tensor = temporal_indices.long().clamp(0, self.ref_tensor.shape[3] - 1).to(self.ref_tensor.device)

            if ref_mode == 'nearest':
                xs = ego_xs if torch.is_tensor(ego_xs) else torch.as_tensor(ego_xs)
                ys = ego_ys if torch.is_tensor(ego_ys) else torch.as_tensor(ego_ys)
                xs = xs.to(self.ref_tensor.device).float().view(-1)
                ys = ys.to(self.ref_tensor.device).float().view(-1)
                t_proj = self._project_index_torch(w_tensor, p_tensor, xs, ys, t_tensor, window)
            else:
                t_proj = t_tensor

            t_spd = torch.clamp(t_proj + 5, max=self.ref_tensor.shape[3] - 1)
            vals = self.ref_tensor[w_tensor, 0, p_tensor, t_proj]
            vals_spd = self.ref_tensor[w_tensor, 0, p_tensor, t_spd]
            rx, ry, rs, rh, rd = vals[:, 0], vals[:, 1], vals_spd[:, 2], vals[:, 3], vals[:, 4]
            return torch.stack([rx, ry, rs, torch.zeros_like(rx), rh, torch.zeros_like(rx),
                                t_proj.float()], dim=-1)

        # CPU 回退路径
        refs = []
        for i in range(len(world_indices)):
            w = world_indices[i]
            a = ego_indices_map[w]
            pid = path_indices[i] if path_indices is not None else 0
            if temporal_indices is not None:
                t = int(temporal_indices[i])
                path = self.candidate_paths[w][a][pid]
                t = max(0, min(t, len(path['pos']) - 1))
                if ref_mode == 'nearest':
                    t = self._project_index_np(path['pos'], float(ego_xs[i]),
                                               float(ego_ys[i]), t, window)
                rx, ry = float(path['pos'][t, 0]), float(path['pos'][t, 1])
                rh, rs = float(path['heading'][t]), float(path['speed'][t])
                nearest = t
            else:
                nearest, (rx, ry, rh, rs) = self._nearest_on_candidate(
                    w, a, pid, float(ego_xs[i]), float(ego_ys[i]))
            refs.append([rx, ry, rs, 0.0, rh, 0.0, float(nearest)])
        return torch.tensor(refs, device=ego_xs.device, dtype=torch.float32)

    def get_road_edges_batch(self, world_indices, ego_xs, ego_ys) -> torch.Tensor:
        """
        批量获取离自车最近的道路边界点，每步只拉一次 road tensor。
        返回 [batch, 2] tensor。
        """
        if self._road_cache is None:
            self._road_cache = self.sim.map_observation_tensor().to_torch().cpu().numpy()
        road_tensor = self._road_cache
        TYPE_IDX = 6
        ROADLINE_TYPE = 1

        edges = []
        for idx in range(len(world_indices)):
            w = world_indices[idx]
            ex = float(ego_xs[idx])
            ey = float(ego_ys[idx])
            road_points = road_tensor[w]
            line_mask = road_points[:, TYPE_IDX] == ROADLINE_TYPE
            line_points = road_points[line_mask]
            if len(line_points) == 0:
                edges.append([ex, ey])
            else:
                dx = line_points[:, 0] - ex
                dy = line_points[:, 1] - ey
                closest = np.argmin(dx * dx + dy * dy)
                edges.append([float(line_points[closest, 0]),
                              float(line_points[closest, 1])])
        return torch.tensor(edges, device=ego_xs.device, dtype=torch.float32)