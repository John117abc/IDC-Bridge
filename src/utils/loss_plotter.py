import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

plt.rcParams.update({'font.sans-serif': ['DejaVu Sans', 'sans-serif']})


class LossPlotter:
    """从训练历史数据生成损失曲线图。"""

    def __init__(self, history_loss, save_dir, prefix="loss"):
        """
        history_loss: list，结构为 [[epoch1_data], [epoch2_data], ...]
                      每 epoch_data 是 [(critic_loss, actor_loss, rho), ...] 的列表
        save_dir: 保存路径目录
        prefix: 文件名前缀
        """
        self.history_loss = history_loss
        self.save_dir = save_dir
        self.prefix = prefix
        os.makedirs(save_dir, exist_ok=True)

    def _flatten(self):
        """展平嵌套 list，返回 (critic_list, actor_list, rho_list) 三个平铺列表。"""
        cl, al, rl = [], [], []
        for epoch_data in self.history_loss:
            if isinstance(epoch_data, (list, tuple)):
                for entry in epoch_data:
                    if isinstance(entry, (list, tuple)) and len(entry) >= 3:
                        cl.append(float(entry[0]))
                        al.append(float(entry[1]) if entry[1] is not None else float('nan'))
                        rl.append(float(entry[2]))
        return np.array(cl), np.array(al), np.array(rl)

    def _add_smoothed(self, ax, x, y, window=50, label="smoothed", **kwargs):
        """在轴上叠加滑动平均线。"""
        if len(y) < window:
            ax.plot(x, y, alpha=0.6, label=label, **kwargs)
            return
        kernel = np.ones(window) / window
        smoothed = np.convolve(y, kernel, mode='valid')
        sx = x[window - 1:]
        ax.plot(sx, smoothed, linewidth=2, label=label, **kwargs)

    def _plot_step_panel(self, data, name, color, window=50):
        """
        画 step 级别的图：上行 linear，下行 log。
        data: 1D numpy array
        name: 图标题前缀，如 "Critic Loss"
        """
        if len(data) == 0:
            return

        x = np.arange(1, len(data) + 1)
        valid = ~np.isnan(data)
        if not valid.any():
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        ax1.plot(x, data, alpha=0.3, linewidth=0.6, color=color, label="raw")
        self._add_smoothed(ax1, x, data, window=window,
                           color=color, label=f"smoothed (w={window})")
        ax1.set_ylabel(f"{name} (linear)")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right', fontsize=8)

        pos = data[valid]
        log_mask = pos > 0
        if log_mask.any():
            log_data = np.log10(np.where(valid & (data > 0), data, np.nan))
            ax2.plot(x, log_data, alpha=0.3, linewidth=0.6, color=color, label="raw")
            self._add_smoothed(ax2, x, log_data, window=window,
                               color=color, label=f"smoothed (w={window})")
        ax2.set_ylabel(f"{name} (log₁₀)")
        ax2.set_xlabel("Training Step")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right', fontsize=8)

        fig.suptitle(f"{name} over Training Steps", fontsize=13, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{self.prefix}_{name.lower().replace(' ','_')}_step_{ts}.png"
        fig.savefig(os.path.join(self.save_dir, fname), dpi=150)
        plt.close(fig)

    def _plot_epoch_aggregate(self, name, color_tuple):
        """每 epoch 聚合四个指标：mean, std, min, max。"""
        agg_data = []
        agg_stds = []
        agg_mins = []
        agg_maxs = []

        for epoch_data in self.history_loss:
            if not isinstance(epoch_data, (list, tuple)):
                continue
            vals = []
            for entry in epoch_data:
                if isinstance(entry, (list, tuple)) and len(entry) >= 3:
                    idx = 0 if name == "Critic Loss" else 1
                    v = float(entry[idx]) if entry[idx] is not None else float('nan')
                    if not np.isnan(v) and v != float('nan'):
                        vals.append(v)
            if vals:
                agg_data.append(np.mean(vals))
                agg_stds.append(np.std(vals))
                agg_mins.append(np.min(vals))
                agg_maxs.append(np.max(vals))

        if not agg_data:
            return

        means = np.array(agg_data)
        stds = np.array(agg_stds)
        mins = np.array(agg_mins)
        maxs = np.array(agg_maxs)

        epochs = np.arange(1, len(means) + 1)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        # ── Linear ──
        ax1.fill_between(epochs, mins, maxs, alpha=0.12, color=color_tuple[0],
                         label="min–max range")
        ax1.fill_between(epochs, means - stds, means + stds, alpha=0.25,
                         color=color_tuple[0], label="±1 std")
        ax1.plot(epochs, means, linewidth=2, color=color_tuple[1],
                 marker='o', markersize=4, label="epoch mean")
        ax1.set_ylabel(f"{name} (linear)")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right', fontsize=8)

        # ── Log ──
        pos_mask = means > 0
        if pos_mask.any():
            log_means = np.where(pos_mask, np.log10(means), np.nan)
            log_stds = np.where(pos_mask, stds / (means * np.log(10)), np.nan)
            log_mins = np.where(pos_mask, np.log10(np.maximum(mins, 1e-10)), np.nan)
            log_maxs = np.where(pos_mask, np.log10(np.maximum(maxs, 1e-10)), np.nan)

            ax2.fill_between(epochs, log_mins, log_maxs, alpha=0.12,
                             color=color_tuple[0], label="min–max range")
            ax2.fill_between(epochs, log_means - log_stds, log_means + log_stds,
                             alpha=0.25, color=color_tuple[0], label="±1 std")
            ax2.plot(epochs, log_means, linewidth=2, color=color_tuple[1],
                     marker='o', markersize=4, label="epoch mean")
        ax2.set_ylabel(f"{name} (log₁₀)")
        ax2.set_xlabel("Epoch")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right', fontsize=8)

        fig.suptitle(f"{name} over Epochs", fontsize=13, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{self.prefix}_{name.lower().replace(' ','_')}_epoch_{ts}.png"
        fig.savefig(os.path.join(self.save_dir, fname), dpi=150)
        plt.close(fig)

    def _plot_rho_step(self, data):
        """ρ 随时间变化，单图 linear。"""
        if len(data) == 0:
            return

        valid = ~np.isnan(data)
        if not valid.any():
            return

        x = np.arange(1, len(data) + 1)

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(x, data, alpha=0.5, linewidth=0.8, color='green', label="ρ (raw)")
        self._add_smoothed(ax, x, data, window=50,
                           color='darkgreen', label="smoothed (w=50)")
        ax.set_ylabel("Penalty ρ")
        ax.set_xlabel("Training Step")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_title("Penalty Coefficient ρ over Training", fontsize=13, fontweight='bold')
        fig.tight_layout()

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{self.prefix}_penalty_rho_{ts}.png"
        fig.savefig(os.path.join(self.save_dir, fname), dpi=150)
        plt.close(fig)

    def plot_all(self):
        """生成全部 5 张损失曲线图。"""
        cl, al, rl = self._flatten()

        if len(cl) == 0 and len(al) == 0:
            print("[LossPlotter] history_loss 为空，跳过绘图。")
            return

        # 1. Critic Loss — step
        if len(cl) > 0 and not np.isnan(cl).all():
            self._plot_step_panel(cl, "Critic Loss", "tab:blue")

        # 2. Actor Loss — step
        if len(al) > 0 and not np.isnan(al).all():
            self._plot_step_panel(al, "Actor Loss", "tab:red")

        # 3. Critic Loss — epoch
        self._plot_epoch_aggregate("Critic Loss", ("lightblue", "tab:blue"))

        # 4. Actor Loss — epoch
        self._plot_epoch_aggregate("Actor Loss", ("lightcoral", "tab:red"))

        # 5. ρ — step
        if len(rl) > 0 and not np.isnan(rl).all():
            self._plot_rho_step(rl)

        print(f"[LossPlotter] 图片已存入 {self.save_dir}")
