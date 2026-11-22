"""画"滚动重规划 + 多模态 + 最终选择"的四联图。

读 idc-eval --planner-dump 产出的 npz（每个 world 一组键：w{w}_steps/cands/sel/tgt/ego）：
    python tools/plot_replan.py --dump /workspace/data/rt_dump.npz \
        --out /workspace/data/rt_replan.png --panels 0,5,10,15
不给 --world 就自动选"重规划次数最多"的世界。
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

COL = ['#d50000', '#2962ff', '#00c853', '#ff6d00', '#aa00ff', '#00bfa5',
       '#795548', '#607d8b']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', default='/workspace/data/rt_dump.npz')
    ap.add_argument('--out', default='/workspace/data/rt_replan.png')
    ap.add_argument('--panels', default='0,5,10,15')
    ap.add_argument('--world', type=int, default=-1)
    ap.add_argument('--window', type=float, default=40.0,
                    help='每个面板以重规划时刻自车为中心的显示半径（m）')
    args = ap.parse_args()
    z = np.load(args.dump, allow_pickle=True)
    worlds = sorted({int(k.split('_')[0][1:]) for k in z.files})
    if not worlds:
        print('dump 里没有数据')
        return
    if args.world >= 0:
        w = args.world
    else:
        w = max(worlds, key=lambda i: len(z[f'w{i}_steps']))
    steps = z[f'w{w}_steps']
    cands = z[f'w{w}_cands']
    sel = z[f'w{w}_sel']
    tgt = z[f'w{w}_tgt']
    ego = z[f'w{w}_ego']
    print('world %d: %d 次重规划, K=%d' % (w, len(steps), cands.shape[1]))

    want = [int(x) for x in args.panels.split(',')]
    fig, axes = plt.subplots(1, len(want), figsize=(5.0 * len(want), 5.2), dpi=130)
    if len(want) == 1:
        axes = [axes]
    for ax, wstep in zip(axes, want):
        ri = None
        for i, s in enumerate(steps):
            if int(s) <= int(wstep):
                ri = i
        if ri is None:
            ax.set_title('step %d：还没有重规划' % wstep)
            continue
        C = np.asarray(cands[ri], np.float32)
        k = C.shape[0]
        for kk in range(k):
            c = C[kk]
            chosen = (int(sel[ri]) == kk)
            ax.plot(c[:, 0], c[:, 1], '-', color=COL[kk % len(COL)],
                    lw=3.2 if chosen else 1.3, alpha=0.95 if chosen else 0.5,
                    label=('cand%d ←选中' % kk) if chosen else ('cand%d' % kk))
        t = min(len(ego), int(wstep) + 1)
        if t > 1:
            ax.plot(ego[:t, 0], ego[:t, 1], 'k-', lw=2.2, label='ego 已走')
        ax.plot(tgt[ri, 0], tgt[ri, 1], 'k*', ms=17, label='重规划时刻自车')
        _r = float(args.window)
        ax.set_xlim(tgt[ri, 0] - _r, tgt[ri, 0] + _r)
        ax.set_ylim(tgt[ri, 1] - _r, tgt[ri, 1] + _r)
        ax.set_aspect('equal')
        ax.grid(alpha=0.25)
        ax.set_title('world %d @ step %d：注入 %d 条，选中 %d'
                     % (w, int(steps[ri]), k, int(sel[ri])), fontsize=10)
        ax.legend(fontsize=7, loc='best')
        ax.set_xlabel('x (m, env)')
    axes[0].set_ylabel('y (m, env)')
    fig.tight_layout()
    fig.savefig(args.out)
    print('saved', args.out)


if __name__ == '__main__':
    main()

