"""离线分析 screen.npz：G0/G1/G2 + PPEF 四分类 + 挽回率 + 图。"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', default='/workspace/data/screen/screen.npz')
    ap.add_argument('--out', default='/workspace/data/screen')
    args = ap.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    z = np.load(args.npz, allow_pickle=True)
    rows = json.loads(str(z['rows']))
    print(f'候选记录 {len(rows)} 条')

    by_scene = {}
    for r in rows:
        by_scene.setdefault(r['scene'], []).append(r)
    for s in by_scene.values():
        s.sort(key=lambda x: x['rank'])
    n = len(by_scene)

    def first_ok(srows, key):
        for r in srows:
            if r[key]:
                return r, True
        return srows[0], False          # 全拒 → 退回 top-1

    res = {g: dict(coll=0, off=0, prog=[], n=0, allfail=0, changed=0)
           for g in ('G0', 'G1', 'G2')}
    ppef_scenes = []
    for sid, srows in by_scene.items():
        g0 = srows[0]
        g1, ok1 = first_ok(srows, 'ok_plan')
        g2, ok2 = first_ok(srows, 'ok_exec')
        for g, r in (('G0', g0), ('G1', g1), ('G2', g2)):
            res[g]['n'] += 1
            res[g]['coll'] += int(r['coll'])
            res[g]['off'] += int(r['off'])
            res[g]['prog'].append(r['prog'])
        if not ok1:
            res['G1']['allfail'] += 1
        if not ok2:
            res['G2']['allfail'] += 1
        if g2['rank'] != g0['rank']:
            res['G2']['changed'] += 1
        # PPEF 场景：top-1 通过 plan 检查，但 exec 失败
        if g0['ok_plan'] and (g0['coll'] or g0['off']):
            ppef_scenes.append((sid, g1, g2))

    print(f'\n=== 场景级（n={n}）===')
    print(f"{'组':4s} {'碰撞':>6s} {'越界':>6s} {'平均进度':>8s} "
          f"{'全拒':>6s} {'改选':>6s}")
    for g in ('G0', 'G1', 'G2'):
        r = res[g]
        print(f'{g:4s} {r["coll"]:6d} {r["off"]:6d} {np.mean(r["prog"]):8.3f} '
              f'{r["allfail"]:6d} {r["changed"]:6d}')

    # 候选级四分类
    pp = sum(1 for r in rows if r['ok_plan'] and not (r['coll'] or r['off']))
    pe = sum(1 for r in rows if r['ok_plan'] and (r['coll'] or r['off']))
    fp = sum(1 for r in rows if not r['ok_plan'] and not (r['coll'] or r['off']))
    fe = sum(1 for r in rows if not r['ok_plan'] and (r['coll'] or r['off']))
    tot = len(rows)
    print(f'\n=== 候选级四分类（共 {tot}）===')
    print(f'  PPEP plan过/exec过   : {pp:6d} ({100*pp/tot:5.1f}%)')
    print(f'  PPEF plan过/exec失败 : {pe:6d} ({100*pe/tot:5.1f}%)  ← 核心')
    print(f'  PFEP plan拒/exec过   : {fp:6d} ({100*fp/tot:5.1f}%)  ← 误杀')
    print(f'  PFEF plan拒/exec失败 : {fe:6d} ({100*fe/tot:5.1f}%)')

    print(f'\n=== PPEF 场景（top-1 plan 通过但执行失败）: '
          f'{len(ppef_scenes)} / {n} = {100*len(ppef_scenes)/n:.1f}% ===')
    if ppef_scenes:
        bad1 = sum(1 for _, g1, _ in ppef_scenes if g1['coll'] or g1['off'])
        bad2 = sum(1 for _, _, g2 in ppef_scenes if g2['coll'] or g2['off'])
        print(f'  G1（规划筛选）仍失败 {bad1}')
        print(f'  G2（推演筛选）仍失败 {bad2}')
        print(f'  挽回 {bad1-bad2} ({100*(bad1-bad2)/max(bad1,1):.0f}%)')

    # 图
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), dpi=160)
    xs = ['G0\ntop-1', 'G1\nplan screen', 'G2\nrollout screen']
    coll = [res[g]['coll'] for g in ('G0', 'G1', 'G2')]
    off = [res[g]['off'] for g in ('G0', 'G1', 'G2')]
    x = np.arange(3)
    axes[0].bar(x - 0.18, coll, 0.34, label='collision', color='#e53935')
    axes[0].bar(x + 0.18, off, 0.34, label='off-road', color='#fb8c00')
    for i in range(3):
        axes[0].text(x[i] - 0.18, coll[i] + 0.3, str(coll[i]), ha='center', fontsize=9)
        axes[0].text(x[i] + 0.18, off[i] + 0.3, str(off[i]), ha='center', fontsize=9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(xs, fontsize=9)
    axes[0].set_ylabel(f'scenes (n={n})')
    axes[0].set_title('Screening: plan-level vs rollout-level')
    axes[0].legend(fontsize=8)
    axes[0].grid(axis='y', alpha=0.25)

    prog = [np.mean(res[g]['prog']) for g in ('G0', 'G1', 'G2')]
    axes[1].bar(x, prog, 0.5, color=['#90a4ae', '#42a5f5', '#66bb6a'])
    for i in range(3):
        axes[1].text(x[i], prog[i] + 0.005, f'{prog[i]:.3f}', ha='center', fontsize=9)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(xs, fontsize=9)
    axes[1].set_ylabel('mean progress')
    axes[1].set_title('Progress (cost of screening)')
    axes[1].grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(f'{args.out}/fig_screen.png')
    plt.close(fig)
    print('\n图已保存', args.out)


if __name__ == '__main__':
    main()
