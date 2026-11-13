"""道路图路径生成诊断脚本。

加载单 world，调用 generate_candidate_paths(use_expert_path=False)，
输出详细的追踪过程日志 + 路径质量诊断图。

用法:
    python src/scripts/utils/test_road_graph_path.py \
        --data-dir /path/to/tfrecords \
        --world-idx 0 \
        [--output-dir diag_output]
"""

import sys
import os
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, '/workspace/idc/src')
from env.env_utils import get_env_config, load_scenes, get_ego_indices
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.dataset import SceneDataLoader
from env.idc_state_builder import GPUDriveObservationBuilder


COL6_LANE = 3
COL6_EDGE = 1
COL6_ROADLINE = 2

COLORS = {
    'bg_lane': '#2ca02c', 'bg_edge': '#d62728', 'bg_line': '#1f77b4',
    'path': '#4daf4a', 'extrap': '#e41a1c', 'ego': '#ff7f0e',
    'gap': '#984ea3', 'start': '#00c853', 'end': '#d50000',
}


def load_env(data_dir, world_idx):
    """加载单个 world 的环境。"""
    num_worlds = world_idx + 1
    data_loader = SceneDataLoader(
        root=data_dir, batch_size=num_worlds, dataset_size=num_worlds,
        sample_with_replacement=False, shuffle=False, seed=42)
    env_config = get_env_config()
    env = GPUDriveTorchEnv(
        config=env_config, data_loader=data_loader,
        max_cont_agents=1, device="cuda", action_type="continuous")
    env.reset()
    return env


def generate_report(world_idx, trace_data, output_path):
    """生成文本诊断报告。"""
    lines = []
    lines.append(f"=== 道路图路径生成诊断 — world {world_idx} ===")

    meta = trace_data.get("world_meta", {})
    entry = trace_data.get("entry", {})
    trace = trace_data.get("trace", [])
    quality = trace_data.get("quality", {})

    # BASIC
    lines.append(f"\nBASIC:")
    lines.append(f"  ego:          ({meta.get('ego_x',0):.1f}, {meta.get('ego_y',0):.1f}) "
                  f"h={meta.get('ego_h_deg',0):.1f}°")
    lines.append(f"  lane_pts:     {meta.get('lane_pts',0)}")
    lines.append(f"  edge_pts:     {meta.get('edge_pts',0)}")
    lines.append(f"  segments:     {meta.get('segments',0)}")

    # ENTRY
    lines.append(f"\nENTRY:")
    lines.append(f"  seg_id:         {entry.get('seg_id','?')}")
    lines.append(f"  entry_idx:      {entry.get('entry_idx','?')} / {entry.get('seg_total_pts','?')}")
    lines.append(f"  seg_length:     {entry.get('seg_length',0):.1f}m")
    lines.append(f"  dist_to_ego:    {entry.get('dist_to_ego',0):.2f}m")
    lines.append(f"  h_diff:         {entry.get('h_diff_deg',0):.1f}°")
    lines.append(f"  residual_pts:   {entry.get('residual_pts',0)}")

    # if residual < 3, flag it
    if entry.get('residual_pts', 99) <= 3:
        lines.append(f"  ⚠ entry near segment END — residual only {entry.get('residual_pts',0)} pts")

    # TRACE
    lines.append(f"\nTRACE ({len(trace)} steps):")
    for si in trace:
        seg_id = si.get('seg_id', '?')
        consumed = si.get('seg_consumed', 0)
        seg_len = si.get('seg_length', 0)
        seg_turn = si.get('seg_turn_deg', 0)
        cum = si.get('cumulated_pts', 0)
        next_seg = si.get('next_seg')
        dead = si.get('dead_end', False)
        end_h = si.get('end_h_deg', 0)

        step_no = si.get('step', 0)
        lines.append(f"  {'─'*55}")
        turn_flag = f" [TURN {abs(seg_turn):.0f}°]" if abs(seg_turn) > 20 else ""
        lines.append(f"  [{step_no}] SEG {seg_id}  ({consumed}pts, {seg_len:.1f}m){turn_flag}")
        lines.append(f"        entry: idx={si.get('seg_entry_idx','?')}  "
                      f"h={si.get('seg_start_h_deg',0):.1f}°")
        lines.append(f"        exit:  h={end_h:.1f}°  "
                      f"end=({si.get('end_x',0):.1f}, {si.get('end_y',0):.1f})")
        lines.append(f"        accumulated: {cum} raw pts")

        candidates = si.get('candidates', [])
        active_cands = [c for c in candidates if not any([
            c.get('skipped_dist'), c.get('skipped_fwd'),
            c.get('skipped_dh'), c.get('skipped_visited')])]
        n_cands = sum(1 for c in candidates if not c.get('skipped_visited'))

        if dead:
            extrap = si.get('extrapolated_pts', 0)
            lines.append(f"        DEAD END — extrapolating {extrap} pts")
            if not active_cands:
                lines.append(f"        → 0 candidates passed filter (all {n_cands} skipped)")
        elif next_seg is not None:
            gap = si.get('next_dist', 0)
            dh = si.get('next_dh_deg', 0)
            lines.append(f"        → NEXT: seg={next_seg}  gap={gap:.1f}m  "
                          f"dh={dh:.1f}°")
            intersection_flag = "[INTERSECTION]" if gap > 5 else ""
            if intersection_flag:
                lines.append(f"          {intersection_flag}")

        # show top candidates (up to 6)
        if active_cands:
            sorted_cands = sorted(active_cands, key=lambda x: -x['score'])
            lines.append(f"        SEARCH: {len(active_cands)} candidates passed")
            for c in sorted_cands[:6]:
                marker = "← BEST" if (next_seg is not None and c['seg_id'] == next_seg
                                       and c['entry_idx'] == si.get('next_idx')) else ""
                lines.append(f"          seg={c['seg_id']:4d}  dist={c['dist']:5.1f}m  "
                              f"fwd={c['fwd']:5.1f}  dh={c['dh_deg']:5.1f}°  "
                              f"score={c['score']:5.2f}  {marker}")
        else:
            n_visited = sum(1 for c in candidates if c.get('skipped_visited'))
            n_total = len(candidates)
            lines.append(f"        SEARCH: 0 passed ({n_visited} visited, {n_total} total)")

    # QUALITY
    lines.append(f"{'─'*55}")
    lines.append(f"\nQUALITY:")
    lines.append(f"  raw trace:         {quality.get('raw_total_pts',0)} pts "
                  f"({quality.get('real_pts',0)} real + {quality.get('extrapolated_pts',0)} extrapolated)")
    lines.append(f"  resampled:         {91} points  "
                  f"length={quality.get('path_length',0):.1f}m")
    ps = quality.get('path_start', (0, 0))
    pe = quality.get('path_end', (0, 0))
    lines.append(f"  path start:        ({ps[0]:.1f}, {ps[1]:.1f})  "
                  f"h={quality.get('path_h_start_deg',0):.1f}°")
    lines.append(f"  path end:          ({pe[0]:.1f}, {pe[1]:.1f})  "
                  f"h={quality.get('path_h_end_deg',0):.1f}°")
    max_gap = quality.get('max_gap', 0)
    gap_ok = max_gap <= 5
    lines.append(f"  max gap:           {max_gap:.1f}m  "
                  f"{'OK ✓' if gap_ok else '⚠ JUMP'}")
    extrap_pct = quality.get('extrapolated_pct', 0)
    if extrap_pct > 50:
        lines.append(f"  extrapolated%:     {extrap_pct:.1f}%  ⚠ >50% extrapolated!")
    else:
        lines.append(f"  extrapolated%:     {extrap_pct:.1f}%")

    # KEY ISSUES
    issues = []
    if quality.get('extrapolated_pct', 0) > 50:
        issues.append(f"⚠ {extrap_pct:.0f}% of path is extrapolated (dead end too early)")
    if entry.get('residual_pts', 99) <= 5:
        issues.append(f"⚠ Entry near segment end: only {entry.get('residual_pts', 0)}"
                       f" pts remaining in first segment")
    if max_gap > 10:
        issues.append(f"⚠ Path jump: inter-point gap {max_gap:.1f}m")
    if quality.get('path_length', 0) < 50:
        issues.append(f"⚠ Path too short: {quality.get('path_length', 0):.0f}m")
    if abs(ps[0]) > 300 or abs(ps[1]) > 300:
        issues.append(f"⚠ Start out of range: ({ps[0]:.0f}, {ps[1]:.0f})")
    if abs(pe[0]) > 300 or abs(pe[1]) > 300:
        issues.append(f"⚠ End out of range: ({pe[0]:.0f}, {pe[1]:.0f})")

    if issues:
        lines.append(f"\nKEY ISSUES:")
        for iss in issues:
            lines.append(f"  {iss}")
    else:
        lines.append(f"\nKEY ISSUES:  None ✓")

    # Write
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    report_text = "\n".join(lines)
    with open(output_path, 'w') as f:
        f.write(report_text)
    print(report_text)
    print(f"\n[report] 已保存: {output_path}")


def plot_diagnostics(world_idx, road_world, path, trace_data, output_path):
    """生成 4 子图诊断图。"""
    fig, axes = plt.subplots(2, 2, figsize=(28, 20))
    ax_all, ax_entry, ax_trace, ax_end = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # --- 全图 ---
    ax_all.set_title(f"(a) Full View — World {world_idx}", fontsize=12)

    # road background
    for col_val, color, label, z in [
        (COL6_EDGE, COLORS['bg_edge'], 'road edge', 0),
        (COL6_ROADLINE, COLORS['bg_line'], 'road line', 1),
        (COL6_LANE, COLORS['bg_lane'], 'lane center', 2),
    ]:
        mask = road_world[:, 6] == col_val
        if mask.any():
            ax_all.scatter(road_world[mask, 0], road_world[mask, 1],
                           c=color, s=1, alpha=0.4, zorder=z, label=label)

    # path
    path_pos = path['pos']
    ax_all.plot(path_pos[:, 0], path_pos[:, 1], '-', color=COLORS['path'],
                linewidth=1.5, label='Generated Path', zorder=5)

    # ego + start/end
    meta = trace_data.get("world_meta", {})
    ax_all.scatter(meta['ego_x'], meta['ego_y'], c=COLORS['ego'], s=100,
                   marker='*', zorder=10, label='Ego Start')
    ax_all.scatter(path_pos[0, 0], path_pos[0, 1], c=COLORS['start'], s=60,
                   marker='o', zorder=10, edgecolors='white')
    ax_all.scatter(path_pos[-1, 0], path_pos[-1, 1], c=COLORS['end'], s=60,
                   marker='s', zorder=10, edgecolors='white')

    ax_all.set_aspect('auto')
    # auto-zoom to path extent with 20% margin
    min_x, max_x = path_pos[:, 0].min(), path_pos[:, 0].max()
    min_y, max_y = path_pos[:, 1].min(), path_pos[:, 1].max()
    margin_x = max(30, (max_x - min_x) * 0.25)
    margin_y = max(30, (max_y - min_y) * 0.25)
    ax_all.set_xlim(min_x - margin_x, max_x + margin_x)
    ax_all.set_ylim(min_y - margin_y, max_y + margin_y)
    ax_all.legend(fontsize=7, markerscale=2)
    ax_all.grid(True, alpha=0.3)

    # --- 入口 zoom ---
    ax_entry.set_title(f"(b) Entry Zoom (50m box)", fontsize=12)
    ex, ey = meta['ego_x'], meta['ego_y']
    margin = 50
    ax_entry.set_xlim(ex - margin, ex + margin)
    ax_entry.set_ylim(ey - margin, ey + margin)

    for col_val, color, z in [
        (COL6_EDGE, COLORS['bg_edge'], 0),
        (COL6_ROADLINE, COLORS['bg_line'], 1),
        (COL6_LANE, COLORS['bg_lane'], 2),
    ]:
        mask = road_world[:, 6] == col_val
        if mask.any():
            ax_entry.scatter(road_world[mask, 0], road_world[mask, 1],
                             c=color, s=3, alpha=0.5, zorder=z)

    ax_entry.plot(path_pos[:, 0], path_pos[:, 1], '-', color=COLORS['path'],
                  linewidth=2, zorder=5, label='Path')
    ax_entry.scatter([ex], [ey], c=COLORS['ego'], s=150, marker='*',
                     zorder=10, label='Ego')
    ax_entry.scatter([path_pos[0, 0]], [path_pos[0, 1]], c=COLORS['start'],
                     s=80, marker='o', zorder=10, edgecolors='white')
    ax_entry.set_aspect('equal')
    ax_entry.legend(fontsize=7)
    ax_entry.grid(True, alpha=0.3)

    # --- 追踪过程 ---
    ax_trace.set_title(f"(c) Trace Steps (colored by visit order)", fontsize=12)

    # road bg (light)
    for col_val, color in [(COL6_EDGE, COLORS['bg_edge']), (COL6_LANE, COLORS['bg_lane'])]:
        mask = road_world[:, 6] == col_val
        if mask.any():
            ax_trace.scatter(road_world[mask, 0], road_world[mask, 1],
                             c=color, s=1, alpha=0.3, zorder=0)

    # lane segments accessed
    trace = trace_data.get("trace", [])
    cmap = plt.get_cmap('tab10')
    seg_xys = {}  # seg_id -> all point coords
    prev_end = None
    for si in trace:
        seg_id = si.get('seg_id')
        seg_length = si.get('seg_length', 0)
        if seg_id is not None and seg_id not in seg_xys and seg_length > 8.0:
            color = cmap(len(seg_xys) % 10)
            ax_trace.text(si.get('end_x', 0), si.get('end_y', 0) + 2,
                          f"S{seg_id}(step{si['step']})", fontsize=7,
                          color=color, ha='center',
                          bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
            seg_xys[seg_id] = color

    # path
    ax_trace.plot(path_pos[:, 0], path_pos[:, 1], '-', color=COLORS['path'],
                  linewidth=1.5, zorder=5, label='Resampled Path')
    ax_trace.scatter([meta['ego_x']], [meta['ego_y']], c=COLORS['ego'], s=80,
                     marker='*', zorder=10, label='Ego')
    ax_trace.set_aspect('auto')
    ax_trace.set_xlim(min_x - margin_x, max_x + margin_x)
    ax_trace.set_ylim(min_y - margin_y, max_y + margin_y)
    ax_trace.legend(fontsize=7)
    ax_trace.grid(True, alpha=0.3)

    # --- 终点 zoom ---
    ax_end.set_title(f"(d) Endpoint Zoom (70m box)", fontsize=12)
    end_x, end_y = path_pos[-1, 0], path_pos[-1, 1]
    ax_end.set_xlim(end_x - 70, end_x + 70)
    ax_end.set_ylim(end_y - 70, end_y + 70)

    for col_val, color in [(COL6_EDGE, COLORS['bg_edge']), (COL6_LANE, COLORS['bg_lane'])]:
        mask = road_world[:, 6] == col_val
        if mask.any():
            ax_end.scatter(road_world[mask, 0], road_world[mask, 1],
                           c=color, s=3, alpha=0.5, zorder=0)

    # distinguish real vs extrapolated
    real_pts = trace_data.get("quality", {}).get("real_pts", 0)
    extrap_pts = trace_data.get("quality", {}).get("extrapolated_pts", 0)
    if extrap_pts > 0:
        total_raw = real_pts + extrap_pts
        # mark extrapolated region on resampled path
        split_ratio = real_pts / total_raw if total_raw > 0 else 1.0
        split_idx = max(1, min(90, int(91 * split_ratio)))
        ax_end.plot(path_pos[:split_idx, 0], path_pos[:split_idx, 1], '-',
                    color=COLORS['path'], linewidth=2, label='Real Trace')
        ax_end.plot(path_pos[split_idx:, 0], path_pos[split_idx:, 1], '--',
                    color=COLORS['extrap'], linewidth=2, label='Extrapolated')

        # draw ego at split point
        ax_end.scatter([path_pos[split_idx, 0]], [path_pos[split_idx, 1]],
                       c=COLORS['extrap'], s=60, marker='x', zorder=10,
                       label=f'Extrap start')
    else:
        ax_end.plot(path_pos[:, 0], path_pos[:, 1], '-', color=COLORS['path'],
                    linewidth=1.5, label='Path')

    ax_end.scatter([path_pos[-1, 0]], [path_pos[-1, 1]], c=COLORS['end'],
                   s=100, marker='s', zorder=10, edgecolors='white', label='End')
    ax_end.set_aspect('equal')
    ax_end.legend(fontsize=7)
    ax_end.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[visual] 诊断图已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="道路图路径生成诊断")
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--world-idx', type=int, default=0)
    parser.add_argument('--output-dir', type=str,
                        default='src/scripts/utils/diag_output')
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    world_idx = args.world_idx

    print(f"=== 道路图路径诊断: world {world_idx} ===")

    # 1. 加载环境
    print("\n[1/5] 加载环境...")
    try:
        env = load_env(args.data_dir, world_idx)
    except Exception as e:
        print(f"错误: {e}")
        return 1

    # 2. 构建 builder
    print("\n[2/5] 构建 builder...")
    scene_files = env.data_batch
    scenes = []
    for f in scene_files:
        with open(f) as fp:
            scenes.append(json.load(fp))
    builder = GPUDriveObservationBuilder(env, scenes)

    ego_indices = get_ego_indices(env, env.num_worlds)

    # 3. 生成路径（带诊断日志）
    print("\n[3/5] 生成道路图路径 (use_expert_path=False)...")
    trace_logs = {}
    builder.generate_candidate_paths(ego_indices, num_paths=1,
                                     use_expert_path=False,
                                     trace_logs_out=trace_logs)

    # 4. 提取结果
    print("\n[4/5] 提取诊断数据...")
    trace_data = trace_logs.get(world_idx, [])
    if not trace_data or len(trace_data) == 0:
        print("  错误: 未获取到追踪日志")
        return 1
    trace_data = trace_data[0]  # list of dicts per world, take first

    a = ego_indices[world_idx]
    path = builder.candidate_paths[world_idx][a][0]

    # 5. 生成输出
    print("\n[5/5] 生成诊断输出...")
    report_path = os.path.join(output_dir, f"rg_path_test_w{world_idx}.txt")
    plot_path = os.path.join(output_dir, f"rg_path_test_w{world_idx}.png")

    generate_report(world_idx, trace_data, report_path)

    # Road tensor for plotting
    road_world = env.sim.map_observation_tensor().to_torch().cpu().numpy()[world_idx]
    plot_diagnostics(world_idx, road_world, path, trace_data, plot_path)

    print(f"\n=== 诊断完成 ===")
    print(f"  报告:   {report_path}")
    print(f"  可视化: {plot_path}")

    del env
    return 0


if __name__ == "__main__":
    sys.exit(main())
