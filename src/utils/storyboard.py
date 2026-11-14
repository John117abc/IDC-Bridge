"""storyboard 风格的可视化：把一次 rollout 画成 GIF / 关键时刻拼图。

设计动机
--------
gpudrive 自带的 ``plot_simulator_state`` 画的是"细线道路 + 小圆点车辆"，
既没有车头朝向，也没有轨迹尾迹，看不出策略是在减速还是绕行；而且它按
``time_steps`` 从已记录轨迹里取帧，取错索引就会得到一张静止图。

这里改成**离线渲染**：eval 只负责把逐帧位姿收进 trace，rollout 结束后再用
matplotlib 画。好处是
  1. rollout 提速（不再每步调用一次重度渲染）；
  2. 画面可读（车头朝向矩形、蓝色尾迹、周车尾迹、专家路径虚线）；
  3. 不会再有"帧索引传错"的坑。
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np                # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

# 靠颜色区分地图语义；未列出的类型统一按 road_edge 处理
TYPE_COLOR = {1: "#cccccc", 2: "#9e9e9e", 3: "#bdbdbd"}


def vehicle_poly(x, y, h, length, width):
    """返回车体四角（世界坐标）。"""
    c, s = np.cos(h), np.sin(h)
    pts = np.array([[length / 2, width / 2], [length / 2, -width / 2],
                    [-length / 2, -width / 2], [-length / 2, width / 2]])
    rot = np.array([[c, -s], [s, c]])
    return pts @ rot.T + np.array([x, y])


class StoryboardPainter:
    """负责把 trace 的某一帧画到给定的 Axes 上。"""

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
        for tp in np.unique(types):
            sel = types == tp
            ax.scatter(m[sel, 0], m[sel, 1], s=0.8,
                       c=TYPE_COLOR.get(int(tp), "#bdbdbd"),
                       linewidths=0, zorder=1)

    def draw(self, ax, ego, partners, info, t, title_prefix=""):
        ax.clear()
        ex, ey, eh = ego[t]
        x0, x1 = ex - self.margin, ex + self.margin
        y0, y1 = ey - self.margin, ey + self.margin
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)

        self._scatter_map(ax, x0, x1, y0, y1)

        if len(self.expert):
            ax.plot(self.expert[:, 0], self.expert[:, 1], ls="--", lw=1.0,
                    c="#525252", alpha=0.9, zorder=2, label="expert reference")

        # 周车尾迹 + 当前车体
        lo = max(0, t - self.trail)
        for tt in range(lo, t + 1):
            alpha = 0.12 + 0.35 * (tt - lo) / max(1, t - lo)
            for p in partners[tt]:
                ax.plot(p[0], p[1], ".", ms=1.3, c="#fd8d3c",
                        alpha=alpha, zorder=2)
        for p in partners[t]:
            gx, gy, gh, _spd = p
            ax.add_patch(Polygon(vehicle_poly(gx, gy, gh, 4.8, 1.9),
                                 closed=True, fc="#fdae6b", ec="#d94801",
                                 lw=0.6, alpha=0.9, zorder=3))

        # ego 尾迹 + 当前车体 + 车头箭头
        ego_lo = max(0, t - self.trail)
        ax.plot(ego[ego_lo:t + 1, 0], ego[ego_lo:t + 1, 1], "-", lw=2.4,
                c="#08519c", zorder=4, label="ego trajectory")
        ax.add_patch(Polygon(vehicle_poly(ex, ey, eh, self.ego_len,
                                          self.ego_wid),
                             closed=True, fc="#2171b5", ec="#08306b",
                             lw=0.9, zorder=5))
        ax.annotate("", xy=(ex + 3.0 * np.cos(eh), ey + 3.0 * np.sin(eh)),
                    xytext=(ex, ey),
                    arrowprops=dict(arrowstyle="-|>", color="#08306b", lw=1.3),
                    zorder=6)

        ev = []
        if info is not None and len(info) > t:
            if info[t][0] > 0:
                ev.append("OFF-ROAD")
            if info[t][1] > 0:
                ev.append("COLLISION")
        ax.set_title(f"{title_prefix}t={t} ({t * 0.1:.1f}s)  "
                     + ("  ".join(ev) if ev else "ok"),
                     fontsize=8,
                     color="#a50f15" if ev else "#252525")
        ax.set_aspect("equal")
        ax.grid(alpha=0.15)
        ax.tick_params(labelsize=6)
        ax.set_xlabel("x (m)", fontsize=6)
        ax.set_ylabel("y (m)", fontsize=6)


def _truncate(ego):
    """agent 被移除后观测是 -1.1e4 之类的哨兵值，截到最后有效步。"""
    ok = (np.abs(ego[:, 0]) < 1e3) & (np.abs(ego[:, 1]) < 1e3)
    idx = np.where(ok)[0]
    return int(idx[-1]) + 1 if len(idx) else 0


def render_world(trace_w, map_pts, expert, out_dir, world, fps=12,
                 ego_len=4.8, ego_wid=2.0, margin=35.0,
                 storyboard=True, gif=True, dpi=110):
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
    subtitle = (f"world {world} | {T} steps | off-road {n_off} | "
                f"collision {n_col} | travelled {dist:.0f} m")

    if storyboard:
        picks = sorted(set([0, T // 5, 2 * T // 5, 3 * T // 5,
                            max(2 * T // 5, 4 * T // 5), T - 1]))
        picks = picks[:6]
        while len(picks) < 6:
            picks.append(T - 1)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        for ax, t in zip(axes.ravel(), picks):
            painter.draw(ax, ego, partners, info, t)
        fig.suptitle(subtitle, fontsize=11)
        fig.tight_layout()
        png_path = os.path.join(out_dir, f"story_world{world}.png")
        fig.savefig(png_path, dpi=dpi)
        plt.close(fig)

    if gif:
        fig, ax = plt.subplots(figsize=(4.8, 4.8))
        frames = []
        for t in range(T):
            painter.draw(ax, ego, partners, info, t)
            fig.tight_layout()
            fig.canvas.draw()
            frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
        plt.close(fig)
        gif_path = os.path.join(out_dir, f"story_world{world}.gif")
        imageio.mimsave(gif_path, frames, fps=fps, loop=0)

    return png_path, gif_path
