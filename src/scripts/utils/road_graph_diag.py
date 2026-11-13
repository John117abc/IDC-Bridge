"""道路图结构诊断脚本 (v2 — col6 vs col8 双列对比)。

在单个 Waymo world 上 dump 所有 map_observation_tensor 点的双类型分布、
车道中心线结构、路口间隙和并行车道情况，输出文本报告 + 可视化图。

用法:
    python src/scripts/utils/road_graph_diag.py \
        --data-dir /path/to/tfrecords \
        --world-idx 0 \
        [--output-dir log/diag]
"""

import sys
import os
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, '/workspace/idc/src')
from env.env_utils import get_env_config
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.datatypes.roadgraph import MapElementIds

# ============================================================
# 列索引 (map_observation_tensor: [W, N, 9])
# ============================================================
COL_X = 0
COL_Y = 1
COL_SEG_LEN = 2
COL_SEG_WIDTH = 3
COL_SEG_HEIGHT = 4
COL_ORIENTATION = 5
COL_GPU_TYPE = 6       # GPUDrive 内部类型 —— 覆盖率 ~100%
COL_ID = 7             # 道路段 ID（同 id 的点属于同一条线段）
COL_VBD_TYPE = 8       # Waymax 兼容类型 —— 覆盖率 ~11%，-1 占 89%

# ============================================================
# Column 6 (GPUDrive 内部类型) 映射
# ============================================================
COL6_TYPE_INFO = {
    0:  ("UNKNOWN/背景",           "#333333"),
    1:  ("ROAD_EDGE(道路边界)",    "#d62728"),
    2:  ("ROAD_LINE(车道标线)",    "#1f77b4"),
    3:  ("LANE_CENTER(车道中心)",  "#2ca02c"),
    4:  ("TYPE_4(CROSSWALK?)",     "#9467bd"),
    5:  ("TYPE_5(SPEED_BUMP?)",    "#8c564b"),
    6:  ("TYPE_6(STOP_SIGN?)",     "#e377c2"),
    7:  ("TYPE_7",                 "#bcbd22"),
    8:  ("TYPE_8",                 "#17becf"),
    9:  ("TYPE_9",                 "#ff7f0e"),
}

COL6_LANE_CENTER_TYPES = {3}   # col6 type 3 是真正的车道中心线
COL6_ROAD_EDGE_TYPES = {1}     # col6 type 1 是道路边界
COL6_ROAD_LINE_TYPES = {2}     # col6 type 2 是车道标线

# ============================================================
# Column 8 (VBD/Waymax 兼容类型) 映射
# ============================================================
VBD_TYPE_INFO = {
    -1: ("UNKNOWN(无标签)",        "#000000"),
    0:  ("LANE_UNDEFINED",          "#222222"),
    1:  ("LANE_FREEWAY",            "#1f77b4"),
    2:  ("LANE_SURFACE_STREET",     "#2ca02c"),
    3:  ("LANE_BIKE_LANE",          "#d62728"),
    5:  ("ROAD_LINE_UNKNOWN",       "#7f7f7f"),
    6:  ("ROAD_LINE_BROKEN_SINGLE_WHITE", "#aaaaaa"),
    7:  ("ROAD_LINE_SOLID_SINGLE_WHITE",  "#999999"),
    8:  ("ROAD_LINE_SOLID_DOUBLE_WHITE",  "#888888"),
    9:  ("ROAD_LINE_BROKEN_SINGLE_YELLOW","#aaaa44"),
    10: ("ROAD_LINE_BROKEN_DOUBLE_YELLOW","#999933"),
    11: ("ROAD_LINE_SOLID_SINGLE_YELLOW", "#888822"),
    12: ("ROAD_LINE_SOLID_DOUBLE_YELLOW", "#777711"),
    13: ("ROAD_LINE_PASSING_DOUBLE_YELLOW","#666600"),
    14: ("ROAD_EDGE_UNKNOWN",        "#8c564b"),
    15: ("ROAD_EDGE_BOUNDARY",       "#e377c2"),
    16: ("ROAD_EDGE_MEDIAN",         "#bcbd22"),
    17: ("STOP_SIGN",                "#ff0000"),
    18: ("CROSSWALK",               "#ff7f0e"),
    19: ("SPEED_BUMP",              "#14cccc"),
    20: ("DRIVEWAY",                "#cc14cc"),
}

VBD_LANE_CENTER_TYPES = {1, 2}
VBD_ROAD_EDGE_TYPES = {14, 15, 16}

# ============================================================
# 环境加载
# ============================================================

def load_single_world(data_dir, world_idx):
    """加载单个 world 的环境，返回 (env, road_tensor_np)。"""
    num_worlds = world_idx + 1
    data_loader = SceneDataLoader(
        root=data_dir,
        batch_size=num_worlds,
        dataset_size=num_worlds,
        sample_with_replacement=False,
        shuffle=False,
        seed=42,
    )
    env_config = get_env_config()
    env = GPUDriveTorchEnv(
        config=env_config,
        data_loader=data_loader,
        max_cont_agents=1,
        device="cuda",
        action_type="continuous",
    )
    env.reset()
    road_tensor = env.sim.map_observation_tensor().to_torch().cpu().numpy()
    road_world = road_tensor[world_idx]
    return env, road_world

# ============================================================
# 类型统计（双列）
# ============================================================

def analyze_types(road_world, col, info_map):
    """统计某一列的类型的点数分布。返回 dict。"""
    counts = {}
    for t_val in np.unique(road_world[:, col]):
        t_int = int(t_val)
        count = int((road_world[:, col] == t_val).sum())
        name = info_map.get(t_int, (f"TYPE_{t_int}", "#888"))
        # handle both 2-tuple and 3-tuple info
        if isinstance(name, tuple) and len(name) >= 2:
            name = name[0]
        elif isinstance(name, tuple):
            name = name[0]
        counts[t_int] = {"name": str(name), "count": count, "pct": count / len(road_world) * 100}
    return counts


def cross_tabulate(road_world):
    """生成 col6 × col8 的交叉表，看看两个系统的对应关系。"""
    col6_vals = road_world[:, COL_GPU_TYPE].astype(int)
    col8_vals = road_world[:, COL_VBD_TYPE].astype(int)

    unique_c6 = sorted(set(col6_vals))
    unique_c8 = sorted(set(col8_vals))

    # 只关注关键的 col8 类型到 col6 的映射
    table = {}
    for c8 in unique_c8:
        mask = col8_vals == c8
        if mask.sum() == 0:
            continue
        c6_for_this_c8 = {}
        for c6 in unique_c6:
            n = int((col6_vals[mask] == c6).sum())
            if n > 0:
                c6_for_this_c8[c6] = n
        vbd_name = VBD_TYPE_INFO.get(c8, (f"VBD_{c8}",))[0]
        table[c8] = {"vbd_name": vbd_name, "total": int(mask.sum()), "col6_types": c6_for_this_c8}
    return table

# ============================================================
# 车道段分组 & 排序
# ============================================================

def group_by_id(pts):
    """按 COL_ID 分组，每组内按前向拓扑排序。返回 {id: np.ndarray[N_i, 9]}"""
    ids = np.unique(pts[:, COL_ID])
    groups = {}
    for gid in ids:
        mask = pts[:, COL_ID] == gid
        group = pts[mask].copy()
        if len(group) > 1:
            order = _sort_polyline_points(group)
            group = group[order]
        groups[int(gid)] = group
    return groups


def _sort_polyline_points(pts):
    """将散乱点排序为连续折线顺序。"""
    n = len(pts)
    if n <= 1:
        return np.arange(n, dtype=int)
    order = [0]
    used = {0}
    while len(order) < n:
        last = pts[order[-1], [COL_X, COL_Y]]
        best_i, best_dist = -1, float("inf")
        for i in range(n):
            if i in used:
                continue
            d = np.hypot(pts[i, COL_X] - last[0], pts[i, COL_Y] - last[1])
            if d < best_dist:
                best_dist, best_i = d, i
        order.append(best_i)
        used.add(best_i)
    return np.array(order, dtype=int)


def analyze_lane_segments(lane_pts):
    """分析车道中心线段：段长、端点、方向等。"""
    groups = group_by_id(lane_pts)
    segments = []
    for gid, pts in groups.items():
        if len(pts) < 2:
            continue
        seg_len = np.sum(np.hypot(np.diff(pts[:, COL_X]), np.diff(pts[:, COL_Y])))
        start_pt = pts[0, [COL_X, COL_Y]]
        end_pt = pts[-1, [COL_X, COL_Y]]
        start_ori = pts[0, COL_ORIENTATION]
        end_ori = pts[-1, COL_ORIENTATION]
        # 统计该段中包含的 col6 类型分布
        col6_types_in_seg = {}
        for t in np.unique(pts[:, COL_GPU_TYPE]).astype(int):
            col6_types_in_seg[int(t)] = int((pts[:, COL_GPU_TYPE] == t).sum())
        segments.append({
            "id": gid,
            "n_points": len(pts),
            "length": float(seg_len),
            "start": start_pt,
            "end": end_pt,
            "start_ori": float(start_ori),
            "end_ori": float(end_ori),
            "col6_types": col6_types_in_seg,
        })
    return segments, groups

# ============================================================
# 路口间隙 & 并行车道检测
# ============================================================

def detect_gaps(segments, max_gap_dist=20.0):
    """检测车道段之间的间隙（路口断裂）。"""
    gaps = []
    for s in segments:
        end_pos = s["end"]
        end_ori = s["end_ori"]
        for t in segments:
            if s["id"] == t["id"]:
                continue
            start_pos = t["start"]
            dist = float(np.hypot(end_pos[0] - start_pos[0], end_pos[1] - start_pos[1]))
            if dist > max_gap_dist:
                continue
            if dist < 0.1:
                continue
            fwd_x = np.cos(end_ori)
            fwd_y = np.sin(end_ori)
            dot = (start_pos[0] - end_pos[0]) * fwd_x + (start_pos[1] - end_pos[1]) * fwd_y
            if dot < -2.0:
                continue
            d_ori = abs(np.arctan2(np.sin(t["start_ori"] - end_ori),
                                   np.cos(t["start_ori"] - end_ori)))
            gaps.append({
                "from_id": s["id"],
                "to_id": t["id"],
                "distance": dist,
                "forward_score": float(dot),
                "heading_diff_deg": float(np.degrees(d_ori)),
            })
    return gaps


def analyze_parallel_lanes(segments, max_heading_diff_deg=25.0, max_lateral_std=4.0):
    """识别并行车道组：方向相近 + 侧向偏移恒定。"""
    groups_found = []
    used = set()
    for i, si in enumerate(segments):
        if si["id"] in used:
            continue
        cluster = [si["id"]]
        used.add(si["id"])
        for j, sj in enumerate(segments):
            if sj["id"] in used:
                continue
            for cid in cluster:
                sc = next(s for s in segments if s["id"] == cid)
                d_ori = abs(np.arctan2(np.sin(sj["start_ori"] - sc["start_ori"]),
                                       np.cos(sj["start_ori"] - sc["start_ori"])))
                if np.degrees(d_ori) > max_heading_diff_deg:
                    continue
                fwd = np.array([np.cos(sc["start_ori"]), np.sin(sc["start_ori"])])
                lateral_at_start = float(np.abs(
                    np.cross(fwd, np.array(sj["start"]) - np.array(sc["start"]))))
                lateral_at_end = float(np.abs(
                    np.cross(fwd, np.array(sj["end"]) - np.array(sc["end"]))))
                lateral_dists = [lateral_at_start, lateral_at_end]
                if np.std(lateral_dists) < max_lateral_std:
                    cluster.append(sj["id"])
                    used.add(sj["id"])
                    break
        if len(cluster) >= 1:
            groups_found.append(cluster)
    return groups_found

# ============================================================
# 可视化 (5 子图)
# ============================================================

def plot_diagnostics(road_world, lane_groups_col6, lane_groups_vbd,
                     segments_col6, segments_vbd,
                     gaps_col6, gaps_vbd,
                     parallel_groups_col6, parallel_groups_vbd,
                     type_counts_col6, type_counts_vbd,
                     output_path):
    """生成 5 子图诊断图：col6 全图 + col8 全图 + col6 车道 + gap + 并行车道。"""
    fig, axes = plt.subplots(2, 3, figsize=(30, 18))
    (ax_c6_all, ax_c8_all, ax_lane), (ax_gap, ax_para, ax_legend) = axes

    cmap_lane = plt.get_cmap('tab20')
    cmap_para = plt.get_cmap('Set1')

    # ---- 图1：col6 (GPUDrive 内部类型) 全图 ----
    ax_c6_all.set_title("(a) All Map Points — Col6 (GPUDrive Types, ~100% coverage)", fontsize=12)
    for t_val in sorted(type_counts_col6.keys()):
        mask = road_world[:, COL_GPU_TYPE] == t_val
        info = COL6_TYPE_INFO.get(int(t_val), (f"T{t_val}", "#888"))
        color = info[1] if isinstance(info, tuple) else "#888"
        name = info[0] if isinstance(info, tuple) else str(info)
        ax_c6_all.scatter(road_world[mask, COL_X], road_world[mask, COL_Y],
                          c=color, s=2, alpha=0.6, label=f"{t_val}:{name}")
    ax_c6_all.set_aspect('equal')
    ax_c6_all.legend(loc='upper right', fontsize=6, ncol=2, markerscale=3)
    ax_c6_all.grid(True, alpha=0.3)

    # ---- 图2：col8 (VBD/Waymax 类型) 全图 ----
    ax_c8_all.set_title("(b) All Map Points — Col8 (VBD Types, ~11% labelled, 89% UNKNOWN)", fontsize=12)
    for t_val in sorted(type_counts_vbd.keys()):
        mask = road_world[:, COL_VBD_TYPE] == t_val
        info = VBD_TYPE_INFO.get(int(t_val), (f"VBD_{t_val}", "#666"))
        color = info[1] if isinstance(info, tuple) else "#666"
        name = info[0] if isinstance(info, tuple) else str(info)
        total = type_counts_vbd[t_val]["count"]
        if total > 0:
            ax_c8_all.scatter(road_world[mask, COL_X], road_world[mask, COL_Y],
                              c=color, s=2, alpha=0.6,
                              label=f"{t_val}:{name}" if total > 50 else None)
    ax_c8_all.set_aspect('equal')
    ax_c8_all.legend(loc='upper right', fontsize=6, ncol=2, markerscale=3)
    ax_c8_all.grid(True, alpha=0.3)

    # ---- 图3：col6 车道中心线 + 方向箭头 ----
    ax_lane.set_title("(c) Lane Centerlines — Col6 (types 1+3) by Segment ID", fontsize=12)
    for i, (gid, pts) in enumerate(lane_groups_col6.items()):
        color = cmap_lane(i % 20)
        ax_lane.plot(pts[:, COL_X], pts[:, COL_Y], '-', color=color, linewidth=1.3,
                     label=f"id={gid}" if i < 15 else None)
        ax_lane.scatter(pts[0, COL_X], pts[0, COL_Y], marker='o', s=25,
                        color=color, edgecolors='white', linewidths=0.5, zorder=5)
        ax_lane.scatter(pts[-1, COL_X], pts[-1, COL_Y], marker='s', s=25,
                        color=color, edgecolors='white', linewidths=0.5, zorder=5)
        for k in range(0, len(pts), max(1, len(pts) // 12)):
            px, py = pts[k, COL_X], pts[k, COL_Y]
            ori = pts[k, COL_ORIENTATION]
            dx, dy = np.cos(ori) * 1.2, np.sin(ori) * 1.2
            ax_lane.arrow(px, py, dx, dy, head_width=0.4, head_length=0.6,
                          fc=color, ec=color, alpha=0.6)
    ax_lane.set_aspect('equal')
    if len(lane_groups_col6) <= 20:
        ax_lane.legend(loc='upper right', fontsize=6, ncol=2)
    ax_lane.grid(True, alpha=0.3)

    # ---- 图4：路口间隙 + 道路边界 ----
    ax_gap.set_title("(d) Gaps & Road Edges — Col6 Lane Segments", fontsize=12)
    for gid, pts in lane_groups_col6.items():
        ax_gap.plot(pts[:, COL_X], pts[:, COL_Y], '-', color='#1f77b4', linewidth=0.8, alpha=0.5)
    # 间隙连线
    for gap in sorted(gaps_col6, key=lambda g: g["distance"])[:25]:
        seg_from = next(s for s in segments_col6 if s["id"] == gap["from_id"])
        seg_to = next(s for s in segments_col6 if s["id"] == gap["to_id"])
        ax_gap.plot([seg_from["end"][0], seg_to["start"][0]],
                     [seg_from["end"][1], seg_to["start"][1]],
                     '--', color='#d62728', linewidth=0.8, alpha=0.6)
        ax_gap.scatter(seg_to["start"][0], seg_to["start"][1],
                       c='#d62728', s=12, zorder=5)
    # 道路边界 (col6 type=2)
    edge_mask = np.isin(road_world[:, COL_GPU_TYPE], list(COL6_ROAD_EDGE_TYPES))
    if edge_mask.any():
        ax_gap.scatter(road_world[edge_mask, COL_X], road_world[edge_mask, COL_Y],
                       c='#bcbd22', s=1, alpha=0.35, label='road edge (col6=2)')
    ax_gap.set_aspect('equal')
    ax_gap.legend(fontsize=7)
    ax_gap.grid(True, alpha=0.3)

    # ---- 图5：并行车道 ----
    ax_para.set_title("(e) Parallel Lane Groups — Col6", fontsize=12)
    for gid, pts in lane_groups_col6.items():
        ax_para.plot(pts[:, COL_X], pts[:, COL_Y], '-', color='#dddddd', linewidth=0.6)
    for gi, cluster in enumerate(parallel_groups_col6):
        if len(cluster) < 2:
            continue
        color = cmap_para(gi % 9)
        for cid in cluster:
            if cid in lane_groups_col6:
                pts = lane_groups_col6[cid]
                ax_para.plot(pts[:, COL_X], pts[:, COL_Y], '-', color=color, linewidth=2.0, alpha=0.8)
                ax_para.scatter(pts[0, COL_X], pts[0, COL_Y], marker='o', s=40,
                                color=color, edgecolors='white', linewidths=0.5, zorder=5)
        if cluster:
            cid0 = cluster[0]
            if cid0 in lane_groups_col6:
                pt0 = lane_groups_col6[cid0][0]
                ax_para.text(pt0[COL_X], pt0[COL_Y] + 3,
                             f'G{gi} ({len(cluster)} lanes)', fontsize=7,
                             color=color, ha='center')
    ax_para.set_aspect('equal')
    ax_para.grid(True, alpha=0.3)

    # ---- 图6：图例面板（交叉表摘要） ----
    ax_legend.axis('off')
    ax_legend.set_title("(f) Col6 ↔ Col8 Cross-Map", fontsize=12, loc='left')

    cross_table = cross_tabulate(road_world)
    legend_lines = []
    legend_lines.append("Col6 type → Col8 mapping (key types):")
    legend_lines.append("")
    for c8 in sorted(cross_table.keys()):
        entry = cross_table[c8]
        vbd_name = entry["vbd_name"]
        c6_dist = entry["col6_types"]
        top_c6 = sorted(c6_dist.items(), key=lambda x: -x[1])[:2]
        c6_str = ", ".join(f"col6={k}({v})" for k, v in top_c6)
        legend_lines.append(f"  VBD {c8:3d} ({vbd_name}): {c6_str}")
        if len(legend_lines) > 25:
            legend_lines.append("  ... (truncated)")
            break
    ax_legend.text(0.02, 0.95, "\n".join(legend_lines),
                   transform=ax_legend.transAxes, fontsize=7,
                   verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[visual] 诊断图已保存: {output_path}")

# ============================================================
# 文本报告
# ============================================================

def format_type_table(type_counts, title, total):
    """格式化类型分布表。"""
    lines = []
    lines.append(f"\n--- {title} ---")
    lines.append(f"{'type':>4}  {'name':<40} {'count':>7}  {'pct':>6}")
    lines.append("-" * 65)
    for t_val in sorted(type_counts.keys()):
        info = type_counts[t_val]
        lines.append(f"{t_val:4d}  {info['name']:<40} {info['count']:7d}  {info['pct']:5.1f}%")
    lines.append(f"\n  总点数: {total}")
    return lines


def format_segment_details(segments, lane_groups, source, total_pts):
    """格式化车道段详情。"""
    lines = []
    n_lane_pts = sum(len(pts) for pts in lane_groups.values())
    lines.append(f"\n--- [{source}] 车道中心线段 ---")
    lines.append(f"  车道中心线总点数: {n_lane_pts}  ({n_lane_pts / total_pts * 100:.1f}%)")
    lines.append(f"  车道段数量: {len(segments)}")

    lengths = [s["length"] for s in segments]
    if lengths:
        lines.append(f"  段长度: min={min(lengths):.1f}m  max={max(lengths):.1f}m  "
                      f"mean={np.mean(lengths):.1f}m  median={np.median(lengths):.1f}m  "
                      f"sum={sum(lengths):.1f}m")

    lines.append(f"\n  各段详情 (前{min(30, len(segments))}段):")
    lines.append(f"  {'id':>6} {'pts':>5} {'len(m)':>8}  {'start°':>8} {'end°':>8}  "
                 f"{'col6_types':>30}")
    lines.append("  " + "-" * 80)
    for s in segments[:30]:
        c6_str = str(s.get("col6_types", {}))
        if len(c6_str) > 30:
            c6_str = c6_str[:27] + "..."
        lines.append(f"  {s['id']:6d} {s['n_points']:5d} {s['length']:8.1f}  "
                      f"{np.degrees(s['start_ori']):8.1f} {np.degrees(s['end_ori']):8.1f}  "
                      f"{c6_str:>30s}")
    return lines


def format_gap_details(gaps, source):
    """格式化间隙检测详情。"""
    lines = []
    lines.append(f"\n--- [{source}] 路口间隙检测 ---")
    lines.append(f"  间隙数量: {len(gaps)}")
    if gaps:
        gap_dists = [g["distance"] for g in gaps]
        gap_oris = [g["heading_diff_deg"] for g in gaps]
        lines.append(f"  距离: min={min(gap_dists):.1f}m  max={max(gap_dists):.1f}m  "
                      f"mean={np.mean(gap_dists):.1f}m")
        lines.append(f"  方向差: min={min(gap_oris):.1f}°  max={max(gap_oris):.1f}°  "
                      f"mean={np.mean(gap_oris):.1f}°")
        lines.append(f"\n  Top 10 间隙 (距离升序):")
        lines.append(f"  {'from':>6} -> {'to':>6}  {'dist':>7}  {'fwd':>7}  {'d°':>7}")
        lines.append("  " + "-" * 45)
        for g in sorted(gaps, key=lambda g: g["distance"])[:10]:
            lines.append(f"  {g['from_id']:6d} -> {g['to_id']:6d}  "
                          f"{g['distance']:7.1f}  {g['forward_score']:7.1f}  "
                          f"{g['heading_diff_deg']:7.1f}")
    return lines


def format_parallel_details(parallel_groups, source):
    """格式化并行车道详情。"""
    lines = []
    lines.append(f"\n--- [{source}] 并行车道分析 ---")
    lines.append(f"  并行车道组数: {len(parallel_groups)}")
    multi = [g for g in parallel_groups if len(g) >= 2]
    single = [g for g in parallel_groups if len(g) < 2]
    lines.append(f"  多车道组 (>=2): {len(multi)}")
    lines.append(f"  单车道: {len(single)}")
    for gi, cluster in enumerate(parallel_groups):
        label = "MULTI" if len(cluster) >= 2 else "single"
        lines.append(f"  group_{gi}: {len(cluster)} lanes [{label}], ids={cluster}")
    return lines


def generate_report(type_counts_col6, type_counts_vbd,
                    segments_col6, lane_pts_col6, lane_groups_col6,
                    segments_vbd, lane_pts_vbd, lane_groups_vbd,
                    gaps_col6, gaps_vbd,
                    parallel_groups_col6, parallel_groups_vbd,
                    cross_table,
                    road_world, report_path):
    """生成双列诊断报告。"""
    total_pts = len(road_world)
    lines = []
    lines.append("=" * 80)
    lines.append("道路图结构诊断报告 (v2 — col6 vs col8 双列对比)")
    lines.append("=" * 80)

    # 0. 快速结论
    col6_lane_pct = len(lane_pts_col6) / total_pts * 100
    col8_lane_pct = len(lane_pts_vbd) / total_pts * 100 if len(lane_pts_vbd) > 0 else 0
    col8_unknown_pct = type_counts_vbd.get(-1, {"pct": 0})["pct"]

    lines.append("\n>>> 关键结论 <<<")
    lines.append(f"  col8 (VBD) 覆盖率: 仅 {100 - col8_unknown_pct:.1f}% ({col8_unknown_pct:.1f}% 为 UNKNOWN=-1)")
    lines.append(f"  col8 车道点占比: {col8_lane_pct:.1f}%  |  col6 车道点占比: {col6_lane_pct:.1f}%")
    lines.append(f"  col6 车道段数: {len(segments_col6)}  |  col8 车道段数: {len(segments_vbd)}")
    lines.append(f"  >>> 路径生成应使用 col6 (GPUDrive 内部类型), 不依赖 col8 <<<")

    # 1. 类型分布 — col6
    lines.extend(format_type_table(type_counts_col6,
                                    "col6 类型分布 (GPUDrive 内部, Column 6)", total_pts))

    # 2. 类型分布 — col8
    lines.extend(format_type_table(type_counts_vbd,
                                    "col8 类型分布 (VBD/Waymax, Column 8)", total_pts))

    # 3. 交叉表
    lines.append("\n--- col6 ↔ col8 交叉映射 ---")
    lines.append(f"  {'col8':>4}  {'vbd_type_name':<38} {'total':>6}  top col6 types")
    lines.append("  " + "-" * 75)
    cross = cross_table
    for c8 in sorted(cross.keys()):
        entry = cross[c8]
        c6_str = ", ".join(f"c6={k}({v})" for k, v in
                           sorted(entry["col6_types"].items(), key=lambda x: -x[1])[:3])
        lines.append(f"  {c8:4d}  {entry['vbd_name']:<38} {entry['total']:6d}  {c6_str}")

    # 4. 车道段 — col6
    lines.extend(format_segment_details(segments_col6, lane_groups_col6, "col6", total_pts))

    # 5. 车道段 — col8 (仅对比摘要)
    if len(lane_pts_vbd) > 0:
        lines.extend(format_segment_details(segments_vbd, lane_groups_vbd, "col8", total_pts))
    else:
        lines.append("\n--- [col8] 车道中心线段: 无数据 ---")

    # 6. 间隙 — col6
    lines.extend(format_gap_details(gaps_col6, "col6"))

    # 7. 间隙 — col8
    lines.extend(format_gap_details(gaps_vbd, "col8"))

    # 8. 并行车道 — col6
    lines.extend(format_parallel_details(parallel_groups_col6, "col6"))

    # 9. 并行车道 — col8
    if parallel_groups_vbd:
        lines.extend(format_parallel_details(parallel_groups_vbd, "col8"))

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    report_text = "\n".join(lines)
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(report_text)
    print(f"\n[report] 文本报告已保存: {report_path}")


def print_col6_summary(type_counts_col6, road_world):
    """打印 col6 类型分布的快速摘要。"""
    print("\n  col6 (GPUDrive 内部) 类型分布:")
    for t_val in sorted(type_counts_col6.keys()):
        info = type_counts_col6[t_val]
        print(f"    col6={t_val:2d} {info['name']:<30s}: {info['count']:6d} ({info['pct']:4.1f}%)")


def print_vbd_summary(type_counts_vbd):
    """打印 col8 类型分布的快速摘要。"""
    print("\n  col8 (VBD) 类型分布:")
    for t_val in sorted(type_counts_vbd.keys()):
        info = type_counts_vbd[t_val]
        print(f"    col8={t_val:2d} {info['name']:<35s}: {info['count']:6d} ({info['pct']:4.1f}%)")


def run_lane_analysis(road_world, lane_types, type_col, source_label):
    """对车道类型集运行完整分析管道。type_col = COL_GPU_TYPE 或 COL_VBD_TYPE。"""
    lane_mask = np.isin(road_world[:, type_col], list(lane_types))
    lane_pts = road_world[lane_mask]
    if len(lane_pts) == 0:
        print(f"  [{source_label}] 无车道点")
        return lane_pts, [], {}, [], []
    segments, groups = analyze_lane_segments(lane_pts)
    gaps = detect_gaps(segments)
    parallel = analyze_parallel_lanes(segments)
    lengths = [s["length"] for s in segments]
    print(f"  [{source_label}] {len(segments)} 段, {len(lane_pts)} 点 "
          f"({len(lane_pts)/len(road_world)*100:.1f}%), "
          f"总长 {sum(lengths):.0f}m, "
          f"min={min(lengths):.1f}m  max={max(lengths):.1f}m")
    return lane_pts, segments, groups, gaps, parallel


def main():
    parser = argparse.ArgumentParser(description="道路图结构诊断工具 (col6 vs col8)")
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--world-idx', type=int, default=0)
    parser.add_argument('--output-dir', type=str, default='src/scripts/utils/diag_output')
    parser.add_argument('--max-gap-dist', type=float, default=20.0)
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    world_idx = args.world_idx

    print(f"=== 道路图诊断 v2: world {world_idx} ===")
    print(f"数据目录: {args.data_dir}")

    # 1. 加载
    print("\n[1/5] 加载环境...")
    try:
        env, road_world = load_single_world(args.data_dir, world_idx)
    except Exception as e:
        print(f"错误: 无法加载环境: {e}")
        return 1
    total_pts = road_world.shape[0]
    print(f"  张量形状: {road_world.shape} ({total_pts} points, 9 features)")

    # 2. 双列类型统计
    print("\n[2/5] 类型统计 (col6 + col8)...")
    type_counts_col6 = analyze_types(road_world, COL_GPU_TYPE, COL6_TYPE_INFO)
    type_counts_vbd = analyze_types(road_world, COL_VBD_TYPE, VBD_TYPE_INFO)
    print_col6_summary(type_counts_col6, road_world)
    print_vbd_summary(type_counts_vbd)

    # 交叉表
    cross_table = cross_tabulate(road_world)
    print("\n  col6 ↔ col8 关键映射 (col8 中点数最多的类型):")
    for c8 in sorted(cross_table.keys(), key=lambda c: -cross_table[c]["total"])[:8]:
        entry = cross_table[c8]
        c6_str = ", ".join(f"c6={k}({v})" for k, v in
                           sorted(entry["col6_types"].items(), key=lambda x: -x[1])[:2])
        print(f"    VBD {c8:3d} → {c6_str}  (共 {entry['total']} 点)")

    # 3. 车道分析 (双列)
    print("\n[3/5] 车道中心线分析...")
    lp_c6, seg_c6, grp_c6, gaps_c6, para_c6 = run_lane_analysis(
        road_world, COL6_LANE_CENTER_TYPES, COL_GPU_TYPE, "col6")
    lp_v8, seg_v8, grp_v8, gaps_v8, para_v8 = run_lane_analysis(
        road_world, VBD_LANE_CENTER_TYPES, COL_VBD_TYPE, "col8")

    # 4. 路口间隙 & 并行车道 (摘要)
    print(f"\n[4/5] 间隙检测:")
    print(f"  col6: {len(gaps_c6)} 个间隙")
    if gaps_c6:
        for g in sorted(gaps_c6, key=lambda g: g["distance"])[:5]:
            print(f"    id {g['from_id']} -> {g['to_id']}: "
                  f"{g['distance']:.1f}m, d_ori={g['heading_diff_deg']:.1f}°")
    print(f"  col8: {len(gaps_v8)} 个间隙")

    print(f"\n  并行车道:")
    multi_c6 = [g for g in para_c6 if len(g) >= 2]
    print(f"  col6: {len(para_c6)} 组, 多车道={len(multi_c6)}")
    multi_v8 = [g for g in para_v8 if len(g) >= 2]
    print(f"  col8: {len(para_v8)} 组, 多车道={len(multi_v8)}")

    # 5. 生成输出
    print("\n[5/5] 生成诊断报告 + 可视化...")
    plot_path = os.path.join(output_dir, f"road_graph_diag_v2_w{world_idx}.png")
    report_path = os.path.join(output_dir, f"road_graph_diag_v2_w{world_idx}.txt")

    plot_diagnostics(road_world, grp_c6, grp_v8,
                     seg_c6, seg_v8,
                     gaps_c6, gaps_v8,
                     para_c6, para_v8,
                     type_counts_col6, type_counts_vbd,
                     plot_path)

    generate_report(type_counts_col6, type_counts_vbd,
                    seg_c6, lp_c6, grp_c6,
                    seg_v8, lp_v8, grp_v8,
                    gaps_c6, gaps_v8,
                    para_c6, para_v8,
                    cross_table,
                    road_world, report_path)

    print(f"\n=== 诊断完成 ===")
    print(f"  可视化: {plot_path}")
    print(f"  报告:    {report_path}")
    del env
    return 0


if __name__ == "__main__":
    sys.exit(main())
