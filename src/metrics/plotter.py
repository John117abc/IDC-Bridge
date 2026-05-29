import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def print_pdms_table(scores_list, logger, total_worlds=None):
    """打印 PDMS 评估表格到终端。

    Args:
        scores_list: list of dict, 每个 dict 是 PDMSScorer.compute() 的输出
        total_worlds: 总 world 数（用于显示 surviving 比例）
    """
    if not scores_list:
        return

    n = len(scores_list)
    avg = {}
    for key in ['driving_score', 'route_completion']:
        avg[key] = np.mean([s[key] for s in scores_list])

    total_collisions = sum(s['counts']['collision_steps'] for s in scores_list)
    total_off_road = sum(s['counts']['off_road_steps'] for s in scores_list)
    total_ddc = sum(s['counts']['ddc_steps'] for s in scores_list)

    nc_zero = sum(1 for s in scores_list if s['penalties']['nc'] == 0.0)
    dac_zero = sum(1 for s in scores_list if s['penalties']['dac'] == 0.0)
    ddc_zero = sum(1 for s in scores_list if s['penalties']['ddc'] == 0.0)

    w_avg = {}
    for key in ['ep', 'ttc', 'comfort', 'lk']:
        w_avg[key] = np.mean([s['weighted'][key] for s in scores_list])

    lines = []
    lines.append("")
    lines.append("=" * 58)
    lines.append("  PDMS Evaluation Results")
    if total_worlds:
        lines.append(f"  Surviving worlds: {n}/{total_worlds}")
    lines.append("=" * 58)
    lines.append(f"  Worlds evaluated: {n}")
    lines.append("-" * 58)
    lines.append(f"  Driving Score            : {avg['driving_score']:.1f} / 100")
    lines.append(f"  Route Completion         : {avg['route_completion']:.1%}")
    lines.append("-" * 58)
    lines.append(f"  Penalties (worlds affected):")
    lines.append(f"    NC  (collision)        : {nc_zero}/{n}  (collision steps: {total_collisions})")
    lines.append(f"    DAC (off-road)         : {dac_zero}/{n}  (off-road steps: {total_off_road})")
    lines.append(f"    DDC (wrong direction)  : {ddc_zero}/{n}  (ddc steps: {total_ddc})")
    lines.append("-" * 58)
    lines.append(f"  Weighted averages:")
    lines.append(f"    EP  (progress)         : {w_avg['ep']:.3f}")
    lines.append(f"    TTC (time-to-collision) : {w_avg['ttc']:.3f}")
    lines.append(f"    C   (comfort)          : {w_avg['comfort']:.3f}")
    lines.append(f"    LK  (lane keeping)     : {w_avg['lk']:.3f}")
    lines.append("-" * 58)
    s80 = sum(1 for s in scores_list if s['driving_score'] >= 80)
    s60 = sum(1 for s in scores_list if 60 <= s['driving_score'] < 80)
    s30 = sum(1 for s in scores_list if 30 <= s['driving_score'] < 60)
    s0 = sum(1 for s in scores_list if s['driving_score'] < 30)
    lines.append(f"  Score distribution:")
    lines.append(f"    ≥80: {s80}  |  60-79: {s60}  |  30-59: {s30}  |  <30: {s0}")
    lines.append("=" * 58)

    for line in lines:
        logger.info(line)


def plot_pdms_radar(scores_list, save_path, title="PDMS Radar"):
    """雷达图：agg 每个 world 的加权项 + 惩罚项标记。

    Args:
        scores_list: list of PDMSScorer.compute() dicts
    """
    if not scores_list:
        return

    # Aggregate
    w_avg = {}
    for key in ['ep', 'ttc', 'comfort', 'lk']:
        w_avg[key] = np.mean([s['weighted'][key] for s in scores_list])

    categories = ['EP\n(Progress)', 'TTC\n(Safety)', 'C\n(Comfort)', 'LK\n(Lane Keep)']
    values = [w_avg['ep'], w_avg['ttc'], w_avg['comfort'], w_avg['lk']]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, values, alpha=0.25, color='steelblue')
    ax.plot(angles, values, color='steelblue', linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax.set_title(title, fontsize=14, pad=20)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def plot_pdms_bar(per_world_scores, save_path, title="Per-World Driving Score"):
    """柱状图：每个 world 的 driving_score。

    Args:
        per_world_scores: list of {world_idx, score, ...}
    """
    if not per_world_scores:
        return

    world_ids = [s.get('world_idx', i) for i, s in enumerate(per_world_scores)]
    scores = [s['driving_score'] for s in per_world_scores]

    fig, ax = plt.subplots(figsize=(max(8, len(scores) * 0.5), 5))
    colors = ['#2ecc71' if s >= 60 else '#f39c12' if s >= 30 else '#e74c3c' for s in scores]
    bars = ax.bar(range(len(scores)), scores, color=colors)
    ax.axhline(y=60, color='green', linestyle='--', alpha=0.5, label='Good (60+)')
    ax.axhline(y=30, color='orange', linestyle='--', alpha=0.5, label='Fair (30+)')
    ax.set_xticks(range(len(scores)))
    ax.set_xticklabels([str(w) for w in world_ids], rotation=45, ha='right', fontsize=8)
    ax.set_ylim(0, 105)
    ax.set_ylabel('Driving Score')
    ax.set_title(title)
    ax.legend()

    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{score:.0f}', ha='center', va='bottom', fontsize=7)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
