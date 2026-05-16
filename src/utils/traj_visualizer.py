import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from utils import get_logger

logger = get_logger('traj_visualizer')

plt.rcParams.update({
    'font.sans-serif': ['DejaVu Sans', 'sans-serif'],
    'axes.unicode_minus': False,
})


COLORS = {
    'expert':        '#2166ac',
    'candidate':     ['#4daf4a', '#ff7f00', '#984ea3'],
    'actual':        '#e41a1c',
    'start':         '#2ca02c',
    'end':           '#d62728',
    'road_line':     '#999999',
    'lane_center':   '#cccccc',
    'road_edge':     '#b15928',
}

PATH_LABELS = ['left (-3.75m)', 'center (0m)', 'right (+3.75m)']


class TrajectoryVisualizer:
    """
    绘制专家轨迹 / Bezier 候选路径 / 实际行驶轨迹 的对比图。
    用于评估 IDC 框架的跟踪效果和路径选择质量。
    """

    def __init__(self, builder, world_idx: int, agent_idx: int):
        self.builder = builder
        self.world_idx = world_idx
        self.agent_idx = agent_idx

        self.actual_x: list = []
        self.actual_y: list = []
        self._roads_cached = None

    # ---- data feeding ----

    def record_step(self, x: float, y: float):
        self.actual_x.append(x)
        self.actual_y.append(y)

    # ---- road background ----

    def _fetch_roads(self):
        if self._roads_cached is not None:
            return self._roads_cached
        rt = self.builder.sim.map_observation_tensor().to_torch().cpu().numpy()
        pts = rt[self.world_idx]  # [R, 9]
        # column indices for map_observation_tensor
        TYPE_IDX = 6
        ROADLINE = 1
        ROADEDGE = 2
        LANECENTER = 3

        self._roads_cached = {
            'road_line':   pts[pts[:, TYPE_IDX] == ROADLINE],
            'road_edge':   pts[pts[:, TYPE_IDX] == ROADEDGE],
            'lane_center': pts[pts[:, TYPE_IDX] == LANECENTER],
        }
        return self._roads_cached

    # ---- main plot ----

    def plot(self, save_path: str = None, title: str = None,
             show_road: bool = True, figsize=(12, 10), dpi=120):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect('equal')

        # --- road ---
        if show_road:
            roads = self._fetch_roads()
            for key, color, lw, alpha in [
                ('road_edge',   COLORS['road_edge'],   0.8, 0.6),
                ('lane_center', COLORS['lane_center'], 0.5, 0.4),
                ('road_line',   COLORS['road_line'],   0.6, 0.5),
            ]:
                pts_arr = roads[key]
                if len(pts_arr) > 0:
                    ax.scatter(pts_arr[:, 0], pts_arr[:, 1],
                               c=color, s=2, alpha=alpha, zorder=0,
                               label=f'road ({key})')

        # --- expert trajectory ---
        expert_pos = self.builder.expert_pos[self.world_idx, self.agent_idx].cpu().numpy()
        ax.plot(expert_pos[:, 0], expert_pos[:, 1],
                color=COLORS['expert'], linewidth=2.0, linestyle='-',
                label='Expert trajectory', zorder=3)

        # --- candidate Bezier paths ---
        paths = self.builder.candidate_paths[self.world_idx][self.agent_idx]
        for i, path in enumerate(paths):
            bp = path['pos']
            ax.plot(bp[:, 0], bp[:, 1],
                    color=COLORS['candidate'][i % len(COLORS['candidate'])],
                    linewidth=1.2, linestyle='--', alpha=0.8,
                    label=f'Candidate {PATH_LABELS[i] if i < len(PATH_LABELS) else i}',
                    zorder=2)

        # --- actual trajectory ---
        if len(self.actual_x) > 0:
            ax.plot(self.actual_x, self.actual_y,
                    color=COLORS['actual'], linewidth=2.0, linestyle='-',
                    marker='.', markersize=2, alpha=0.9,
                    label='Actual trajectory', zorder=5)

        # --- start / end ---
        sx, sy = float(expert_pos[0, 0]), float(expert_pos[0, 1])
        ex, ey = float(expert_pos[-1, 0]), float(expert_pos[-1, 1])
        ax.scatter(sx, sy, c=COLORS['start'], s=80, marker='o', zorder=6,
                   edgecolors='white', linewidths=0.5, label='Start')
        ax.scatter(ex, ey, c=COLORS['end'], s=80, marker='s', zorder=6,
                   edgecolors='white', linewidths=0.5, label='End')

        # --- heading arrows at start / end ---
        h0 = float(self.builder.expert_heading[self.world_idx, self.agent_idx, 0].item())
        if len(self.actual_x) > 0:
            ha = float(self.builder.expert_heading[self.world_idx, self.agent_idx,
                                                    min(len(self.actual_x) - 1,
                                                        self.builder.EXPERT_TRAJ_LEN - 1)].item())
        else:
            ha = h0
        arrow_len = 2.5
        self._draw_arrow(ax, sx, sy, h0, arrow_len, COLORS['start'])
        self._draw_arrow(ax, ex, ey, ha, arrow_len, COLORS['end'])

        # --- decoration ---
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(title or f'Trajectory Comparison — world {self.world_idx}')
        ax.legend(loc='upper left', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)

    def save_plot(self, base_dir: str, epoch: int) -> str:
        """封装保存：自动创建日期子目录 + 时间戳文件名。"""
        logger.debug(f'自动创建日期子目录 + 时间戳文件名')
        now = datetime.now()
        date_dir = os.path.join(base_dir, now.strftime('%Y%m%d'))
        os.makedirs(date_dir, exist_ok=True)

        filename = (f'epoch_{epoch:03d}_world_{self.world_idx:02d}_'
                    f'{now.strftime("%H%M%S")}.png')
        save_path = os.path.join(date_dir, filename)

        self.plot(save_path=save_path,
                  title=f'Epoch {epoch:03d} | World {self.world_idx}')
        return save_path

    @staticmethod
    def _draw_arrow(ax, x, y, heading, length, color):
        dx = length * np.cos(heading)
        dy = length * np.sin(heading)
        ax.annotate('', xy=(x + dx, y + dy), xytext=(x, y),
                    arrowprops=dict(arrowstyle='->', color=color,
                                    lw=2.0, alpha=0.7))
