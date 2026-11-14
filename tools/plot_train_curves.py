"""从训练日志解析 PDMS / 损失 / 跟踪指标并出图。

用法：
    python tools/plot_train_curves.py --log logs/overnight.log --out plots/
日志来源可以是 ``tools/fetch_remote.py`` 拉下来的副本。
"""
import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np                # noqa: E402

RE_EPOCH = re.compile(r"回合\s+(\d+)/\d+,\s*步数\s+(\d+)/")
RE_PDMS = re.compile(
    r"\[PDMS\] score=([\d.]+) completion=([\d.]+)% collisions=(\d+) "
    r"off_road=(\d+) \(surviving: (\d+)/(\d+)\)")
RE_LOSS = re.compile(r"Critic Loss:\s*([\d.]+|nan|\s*-+\s*),.*?"
                     r"Actor Loss:\s*([\d.]+|nan|\s*-+\s*),.*?"
                     r"rho\):\s*([\d.]+)")
RE_DIAG = re.compile(r"\[DIAG-ref\].*?median=([\d.]+) p90=([\d.]+).*?"
                     r"frac<5m=([\d.]+)%.*?idx_lag_med=(-?[\d.]+).*?"
                     r"good=(\d+)")


def _num(s):
    s = s.strip()
    if s in ("-", "--", "nan") or set(s) <= {"-"}:
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def parse_log(path):
    pdms, loss, diag = [], [], []
    epoch = 0
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = RE_EPOCH.search(line)
            if m:
                epoch = int(m.group(1))
                continue
            m = RE_PDMS.search(line)
            if m:
                pdms.append((epoch,) + tuple(float(x) for x in m.groups()))
                continue
            m = RE_DIAG.search(line)
            if m:
                diag.append((epoch,) + tuple(float(x) for x in m.groups()))
                continue
            m = RE_LOSS.search(line)
            if m:
                loss.append((epoch, _num(m.group(1)), _num(m.group(2)),
                             float(m.group(3))))
    P = np.asarray(pdms, dtype=float)
    L = np.asarray(loss, dtype=float)
    D = np.asarray(diag, dtype=float)
    return P, L, D


def per_epoch(A, cols):
    e = A[:, 0].astype(int)
    ue = np.unique(e)
    out = []
    for c in cols:
        v = A[:, c]
        ok = np.isfinite(v)
        out.append(np.array([np.nanmean(v[ok & (e == k)]) if np.any(ok & (e == k))
                             else np.nan for k in ue]))
    return ue, out


def smooth(y, k=25):
    y = np.asarray(y, dtype=float)
    if len(y) < k:
        return y
    ker = np.ones(k) / k
    pad = np.pad(y, (k // 2, k - 1 - k // 2), mode="edge")
    return np.convolve(pad, ker, mode="valid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="训练日志路径")
    ap.add_argument("--out", default="plots", help="输出目录")
    ap.add_argument("--peak-window", type=int, default=25)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    P, L, D = parse_log(args.log)
    if not len(P):
        raise SystemExit("日志里没解析到 [PDMS] 行，检查 --log 路径")
    print(f"解析: PDMS {len(P)} 条, loss {len(L)} 条, diag {len(D)} 条, "
          f"轮次 1~{int(P[-1, 0])}")

    ep_p, (score, comp, coll, off, surv) = per_epoch(P, [1, 2, 3, 4, 5])
    ep_l, (closs, aloss, rho) = per_epoch(L[L[:, 0] > 0], [1, 2, 3]) if len(L) else (
        np.array([]), [[], [], []])
    ep_d, (dmed, dp90, df5, _lag) = per_epoch(D[D[:, 0] > 0], [1, 2, 3, 4]) if len(D) else (
        np.array([]), [[], [], [], []])

    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                         "figure.dpi": 130})
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.6))

    ax = axes[0, 0]
    ax.plot(ep_p, score, color="#9ecae1", lw=0.8, alpha=0.8)
    l1, = ax.plot(ep_p, smooth(score), color="#08519c", lw=2,
                  label="PDMS score (avg)")
    ax2 = ax.twinx()
    l2, = ax2.plot(ep_p, smooth(comp), color="#e6550d", lw=1.6,
                   label="route completion %")
    ax.set_xlabel("epoch"); ax.set_ylabel("PDMS score", color="#08519c")
    ax2.set_ylabel("completion %", color="#e6550d", labelpad=28)
    ax2.set_ylim(0, 100)
    ax.set_title("PDMS score & route completion\n(light = per-epoch raw)")
    ax.grid(False); ax2.grid(False)
    ax.legend(handles=[l1, l2], loc="lower right", fontsize=8)

    ax = axes[0, 1]
    ax.plot(ep_p, smooth(coll), color="#a50f15", lw=1.8, label="collisions (steps/ep)")
    ax.plot(ep_p, smooth(off), color="#6a51a3", lw=1.8, label="off-road (steps/ep)")
    ax.set_xlabel("epoch"); ax.set_ylabel("steps per episode (sum over worlds)")
    ax.legend(loc="upper right"); ax.set_title("safety: collisions / off-road")

    ax = axes[1, 0]
    if len(ep_l):
        ax.plot(ep_l, closs, color="#d9d9d9", lw=0.6)
        ax.plot(ep_l, smooth(closs), color="#252525", lw=1.8, label="critic loss")
        ax.plot(ep_l, aloss, color="#c6dbef", lw=0.6, alpha=0.8)
        ax.plot(ep_l, smooth(aloss), color="#2171b5", lw=1.8, label="actor loss")
        ax.set_yscale("log")
        axb = ax.twinx()
        lb, = axb.plot(ep_l, smooth(rho), color="#31a354", lw=1.4,
                       label=r"$\rho$")
        axb.set_ylabel(r"penalty factor $\rho$", color="#31a354", labelpad=20)
        ax.grid(False); axb.grid(False)
        hh, _ = ax.get_legend_handles_labels()
        ax.legend(handles=hh + [lb], loc="lower left", fontsize=8)
    ax.set_xlabel("epoch"); ax.set_ylabel("loss (log)")
    ax.set_title("critic / actor loss")

    ax = axes[1, 1]
    if len(ep_d):
        ax.plot(ep_d, smooth(df5), color="#238b45", lw=1.8,
                label="worlds within 5 m of reference (%)")
        ax.set_ylim(0, 100)
        axd = ax.twinx()
        lm, = axd.plot(ep_d, smooth(dmed), color="#2171b5", lw=1.6,
                       label="median pos_err (m)")
        lp, = axd.plot(ep_d, smooth(dp90), color="#6baed6", lw=1.2, ls="--",
                       label="p90 pos_err (m)")
        axd.set_yscale("log"); axd.set_ylim(0.1, 500)
        axd.set_ylabel("tracking error (m, log)", color="#2171b5", labelpad=22)
        ax.grid(False); axd.grid(False)
        hh, _ = ax.get_legend_handles_labels()
        ax.legend(handles=hh + [lm, lp], loc="center right", fontsize=8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("frac of worlds with pos_err < 5 m (%)", color="#238b45")
    ax.set_title("tracking quality")

    fig.tight_layout()
    p = os.path.join(args.out, "train_summary.png")
    fig.savefig(p); plt.close(fig)
    print("saved", p)

    # 分段均值表
    def seg(y, a, b):
        m = (ep_p >= a) & (ep_p <= b)
        v = np.asarray(y)[m]
        v = v[np.isfinite(v)]
        return float(np.mean(v)) if len(v) else float("nan")

    last = int(ep_p[-1])
    print("\n轮次区间        PDMS   completion%  collisions  off_road  surviving")
    for a in range(1, max(2, last), max(50, last // 12)):
        b = min(a + max(50, last // 12) - 1, last)
        print(f"{a:5d}-{b:<6d} {seg(score, a, b):7.1f} {seg(comp, a, b):12.1f} "
              f"{seg(coll, a, b):11.1f} {seg(off, a, b):9.1f} {seg(surv, a, b):9.2f}")


if __name__ == "__main__":
    main()
