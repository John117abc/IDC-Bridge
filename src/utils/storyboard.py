"""storyboard 风格的可视化：把一次 rollout 画成 GIF / 关键时刻拼图。

设计动机
--------
gpudrive 自带的 ``plot_simulator_state`` 画的是"细线道路 + 小圆点车辆"，
既没有车头朝向，也没有轨迹尾迹，看不出策略是在减速还是绕行；而且它按
``time_steps`` 从已记录轨迹里取帧，取错索引就会得到一张静止图。

这里改成**离线渲染**：eval 只负责把逐帧位姿收进 trace，rollout 结束后再用
matplotlib 画。好处是
  1. rollout 提速（不再每步调用一次重度渲染）；
  2. 画面可读（车头朝向矩形、尾迹、图例、醒目的参考轨迹、加粗的车道边界）；
  3. 不会再有"帧索引传错"的坑。

地图点云的类型编号（沿用 traj_visualizer.py）
    1 = road_edge（车道边界）  2 = road_line  3 = lane_center
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np                # noqa: E402
from matplotlib.lines import Line2D     # noqa: E402
from matplotlib.patches import Patch, Polygon  # noqa: E402

ROAD_EDGE, ROAD_LINE, LANE_CENTER = 1, 2, 3

# 语义 -> (颜色, 点大小, 透明度, 图例名)
ROAD_STYLE = {
    ROAD_EDGE: ("#7f3b08", 6.0, 0.95, "lane boundary (road_edge)"),
    ROAD_LINE: ("#8c8c8c", 3.4, 0.90, "road marking (road_line)"),
    LANE_CENTER: ("#c8c8c8", 2.4, 0.80, "lane center"),
}
ROAD_OTHER = ("#d0d0d0", 1.6, 0.60, "other road points")

C_EXPERT = "#d01c8b"      # 参考轨迹：品红，最醒目
C_EGO = "#08519c"
C_EGO_FILL = "#2b8cbe"
C_NB = "#fdae6b"
C_NB_EDGE = "#d94801"
C_NB_TRAIL = "#f16913"


def vehicle_poly(x, y, h, length, width):
    """返回车体四角（世界坐标）。"""
    c, s = np.cos(h), np.sin(h)
    pts = np.array([[length / 2, width / 2], [length / 2, -width / 2],
                    [-length / 2, -width / 2], [-length / 2, width / 2]])
    rot = np.array([[c, -s], [s, c]])
    return pts @ rot.T + np.array([x, y])


def legend_handles():
    """统一的图例（顺序固定，方便对照）。"""
    return [
        Line2D([], [], color=C_EXPERT, ls="--", lw=2.4, label="expert reference"),
        Line2D([], [], color=C_EGO, lw=2.6, label="ego trajectory"),
        Patch(facecolor=C_EGO_FILL, edgecolor=C_EGO, label="ego vehicle"),
        Patch(facecolor=C_NB, edgecolor=C_NB_EDGE, label="neighbor vehicle"),
        Line2D([], [], color=C_NB_TRAIL, ls=":", lw=1.4, label="neighbor trail"),
        Line2D([], [], color=ROAD_STYLE[ROAD_EDGE][0], ls="none", marker="s",
               ms=5, label=ROAD_STYLE[ROAD_EDGE][3]),
        Line2D([], [], color=ROAD_STYLE[ROAD_LINE][0], ls="none", marker="s",
               ms=3.2, label=ROAD_STYLE[ROAD_LINE][3]),
        Line2D([], [], color=ROAD_STYLE[LANE_CENTER][0], ls="none", marker="s",
               ms=2.4, label=ROAD_STYLE[LANE_CENTER][3]),
    ]


class StoryboardPainter:
    """把 trace 的某一帧画到给定 Axes 上。"""

    def __init__(self, map_pts, expert, ego_len=4.8, ego_wid=2.0,
                 margin=35.0, trail=40):
        self.map_pts = np.asarray(map_pts) if map_pts is not None else np.zeros((0, 7))
        self.expert = np.asarray(expert) if expert is not None else np.zeros((0, 2))
        self.ego_len = float(ego_len)
        self.ego_wid = float(ego_wid)
        self.margin = float(margin)
        self.trail = int(trail)

    def _scatter_map(self, ax, x0, x1, y0, y1):
        if not len(self.map_pts):
            return
        m = self.map_pts
        keep = (m[:, 0] > x0) & (m[:, 0] < x1) & (m[:, 1] > y0) & (m[:, 1] < y1)
        m = m[keep]
        if not len(m):
            return
        types = m[:, 6] if m.shape[1] > 6 else np.zeros(len(m))
        # 先画点云，再单独把 road_edge 加粗覆盖一层，保证车道边界清晰
        for tp in np.unique(types):
            color, size, alpha, _ = ROAD_STYLE.get(int(tp), ROAD_OTHER)
            sel = types == tp
            ax.scatter(m[sel, 0], m[sel, 1], s=size, c=color, alpha=alpha,
                       linewidths=0, zorder=1)
        sel = types == ROAD_EDGE
        if np.any(sel):
            ax.scatter(m[sel, 0], m[sel, 1], s=11.0, c=ROAD_STYLE[ROAD_EDGE][0],
                       alpha=0.95, linewidths=0, zorder=1.5)

    def draw(self, ax, ego, partners, info, t, legend=False, title_prefix=""):
        ax.clear()
        ex, ey, eh = ego[t]
        x0, x1 = ex - self.margin, ex + self.margin
        y0, y1 = ey - self.margin, ey + self.margin
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)

        self._scatter_map(ax, x0, x1, y0, y1)

        # 专家参考轨迹：加粗品红虚线。zorder 放在 ego 尾迹之上，
        # 否则两者重合时参考线会被蓝色尾迹完全盖住，看不出跟踪贴合程度。
        if len(self.expert):
            ax.plot(self.expert[:, 0], self.expert[:, 1], ls="--", lw=3.0,
                    c=C_EXPERT, alpha=0.95, zorder=6,
                    marker="o", ms=2.2, markevery=8)

        # 周车尾迹 + 当前车体
        lo = max(0, t - self.trail)
        for tt in range(lo, t + 1):
            alpha = 0.15 + 0.45 * (tt - lo) / max(1, t - lo)
            for p in partners[tt]:
                ax.plot(p[0], p[1], ".", ms=1.6, c=C_NB_TRAIL,
                        alpha=alpha, zorder=2.5)
        for p in partners[t]:
            gx, gy, gh, _spd = p
            ax.add_patch(Polygon(vehicle_poly(gx, gy, gh, 4.8, 1.9),
                                 closed=True, fc=C_NB, ec=C_NB_EDGE,
                                 lw=1.0, alpha=0.95, zorder=4))

        # ego 尾迹 + 当前车体 + 车头箭头
        ego_lo = max(0, t - self.trail)
        ax.plot(ego[ego_lo:t + 1, 0], ego[ego_lo:t + 1, 1], "-", lw=2.8,
                c=C_EGO, zorder=5)
        ax.add_patch(Polygon(vehicle_poly(ex, ey, eh, self.ego_len,
                                          self.ego_wid),
                             closed=True, fc=C_EGO_FILL, ec="#08306b",
                             lw=1.2, zorder=7))
        ax.annotate("", xy=(ex + 3.4 * np.cos(eh), ey + 3.4 * np.sin(eh)),
                    xytext=(ex, ey),
                    arrowprops=dict(arrowstyle="-|>", color="#08306b", lw=1.6),
                    zorder=8)

        ev = []
        if info is not None and len(info) > t:
            if info[t][0] > 0:
                ev.append("OFF-ROAD")
            if info[t][1] > 0:
                ev.append("COLLISION")
        ax.set_title(f"{title_prefix}t={t} ({t * 0.1:.1f}s)  "
                     + ("  ".join(ev) if ev else "ok"),
                     fontsize=10, fontweight="bold",
                     color="#a50f15" if ev else "#252525")
        ax.set_aspect("equal")
        ax.grid(alpha=0.18)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("x (m)", fontsize=8)
        ax.set_ylabel("y (m)", fontsize=8)

        if legend:
            ax.legend(handles=legend_handles(), loc="upper right",
                      fontsize=6.8, framealpha=0.92, borderpad=0.5,
                      labelspacing=0.35, ncol=1)


def _truncate(ego):
    """agent 被移除后观测是 -1.1e4 之类的哨兵值，截到最后有效步。"""
    ok = (np.abs(ego[:, 0]) < 1e3) & (np.abs(ego[:, 1]) < 1e3)
    idx = np.where(ok)[0]
    return int(idx[-1]) + 1 if len(idx) else 0


def render_world(trace_w, map_pts, expert, out_dir, world, fps=12,
                 ego_len=4.8, ego_wid=2.0, margin=35.0,
                 storyboard=True, gif=True, dpi=120):
    """为一个 world 输出 storyboard PNG + GIF。返回 (png_path, gif_path)。"""
    import imageio

    ego = np.asarray(trace_w["ego"], dtype=np.float32)
    info = np.asarray(trace_w["info"], dtype=np.float32)
    partners = trace_w["partners"]
    T = _truncate(ego)
    if T < 2:
        print(f"[storyboard] world {world}: 有效帧不足，跳过")
        return None, None
    ego, info, partners = ego[:T], info[:T], partners[:T]

    painter = StoryboardPainter(map_pts, expert, ego_len=ego_len,
                                ego_wid=ego_wid, margin=margin)
    os.makedirs(out_dir, exist_ok=True)
    png_path = gif_path = None

    dist = float(np.sum(np.linalg.norm(np.diff(ego[:, :2], axis=0), axis=1)))
    n_off = int(np.sum(info[:, 0] > 0))
    n_col = int(np.sum(info[:, 1] > 0))
    subtitle = (f"world {world}  |  {T} steps  |  off-road {n_off}  |  "
                f"collision {n_col}  |  travelled {dist:.0f} m")

    if storyboard:
        picks = sorted(set([0, T // 5, 2 * T // 5, 3 * T // 5,
                            max(2 * T // 5, 4 * T // 5), T - 1]))
        picks = (picks + [T - 1] * 6)[:6]
        fig, axes = plt.subplots(2, 3, figsize=(16, 10.5))
        for ax, t in zip(axes.ravel(), picks):
            painter.draw(ax, ego, partners, info, t)
        fig.legend(handles=legend_handles(), loc="lower center", ncol=4,
                   fontsize=9, framealpha=0.95, bbox_to_anchor=(0.5, 0.0))
        fig.suptitle(subtitle, fontsize=12, fontweight="bold")
        fig.tight_layout(rect=(0, 0.045, 1, 1))
        png_path = os.path.join(out_dir, f"story_world{world}.png")
        fig.savefig(png_path, dpi=dpi)
        plt.close(fig)

    if gif:
        fig, ax = plt.subplots(figsize=(5.4, 5.4), dpi=90)
        frames = []
        for t in range(T):
            painter.draw(ax, ego, partners, info, t, legend=True)
            fig.tight_layout()
            fig.canvas.draw()
            frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
        plt.close(fig)
        gif_path = os.path.join(out_dir, f"story_world{world}.gif")
        imageio.mimsave(gif_path, frames, fps=fps, loop=0)

    return png_path, gif_path
