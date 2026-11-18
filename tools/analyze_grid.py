"""离线重算 grid2 结果（修正 exp_grid.py 里的解包 bug）。

grid2.npz 里保留的可靠字段：okp_cv/okp_tr/oke_cv/oke_tr/off/prog。
exec 失败 = not oke_tr（= 碰撞 or 越界，判据已对齐）；碰撞 = exec 失败 且 非越界。
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', default='/workspace/data/grid2/grid2.npz')
    ap.add_argument('--out', default='/workspace/data/grid2')
    args = ap.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    z = np.load(args.npz, allow_pickle=True)
    rows = json.loads(str(z['rows']))
    print(f'候选 {len(rows)} 条')
    r0 = rows[0]
    print('字段检查:', {k: r0[k] for k in ('rank', 'oke_tr', 'off', 'coll')})

    for r in rows:
        r['exec_fail'] = not bool(r['oke_tr'])
        r['off'] = bool(r['off'])
        r['coll'] = bool(r['exec_fail'] and not r['off'])
        r['bad'] = bool(r['coll'] or r['off'])

    by = {}
    for r in rows:
        by.setdefault(r['scene'], []).append(r)
    for v in by.values():
        v.sort(key=lambda x: x['rank'])
    n = len(by)

    POL = [('G0', None), ('G1cv', 'okp_cv'), ('G1true', 'okp_tr'),
           ('G2cv', 'oke_cv'), ('G2true', 'oke_tr')]

    def first_ok(srows, key):
        for r in srows:
            if r[key]:
                return r
        return srows[0]

    S = {}
    for name, key in POL:
        S[name] = {sid: (srows[0] if key is None else first_ok(srows, key))
                   for sid, srows in by.items()}
    base = S['G0']

    print(f'\n===== 场景级（n={n}）=====')
    print(f"{'策略':8s} {'碰撞':>5s} {'越界':>5s} {'合计失败':>7s} {'全拒':>5s} "
          f"{'救回':>5s} {'新增':>5s} {'改选':>5s} {'进度':>7s}")
    summ = {}
    for name, key in POL:
        sel = S[name]
        coll = sum(int(r['coll']) for r in sel.values())
        off = sum(int(r['off']) for r in sel.values())
        fail = sum(int(r['bad']) for r in sel.values())
        allf = 0 if key is None else sum(
            1 for srows in by.values() if not any(x[key] for x in srows))
        rescue = new = changed = 0
        for sid, r in sel.items():
            b = base[sid]
            if b['bad'] and not r['bad']:
                rescue += 1
            if not b['bad'] and r['bad']:
                new += 1
            if r['rank'] != b['rank']:
                changed += 1
        prog = float(np.mean([r['prog'] for r in sel.values()]))
        summ[name] = dict(coll=coll, off=off, fail=fail, allf=allf,
                          rescue=rescue, new=new, changed=changed, prog=prog)
        print(f'{name:8s} {coll:5d} {off:5d} {fail:7d} {allf:5d} '
              f'{rescue:5d} {new:5d} {changed:5d} {prog:7.3f}')

    print(f'\n===== 关键对比 =====')
    d1 = summ['G2cv']['coll'] - summ['G1cv']['coll']
    d2 = summ['G2true']['coll'] - summ['G1true']['coll']
    print(f'  碰撞: G2cv - G1cv = {d1:+d}   |   G2true - G1true = {d2:+d}')
    print(f'  越界: G2cv - G1cv = {summ["G2cv"]["off"]-summ["G1cv"]["off"]:+d}'
          f'   |   G2true - G1true = '
          f'{summ["G2true"]["off"]-summ["G1true"]["off"]:+d}')
    print(f'  碰撞: G1cv - G0 = {summ["G1cv"]["coll"]-summ["G0"]["coll"]:+d}'
          f'   |   G2cv - G0 = {summ["G2cv"]["coll"]-summ["G0"]["coll"]:+d}')

    print(f'\n===== 候选级判据翻转 =====')
    def flips(a, b):
        up = sum(1 for r in rows if not r[a] and r[b])
        dn = sum(1 for r in rows if r[a] and not r[b])
        return up, dn
    for label, a, b in [('预测 plan', 'okp_cv', 'okp_tr'),
                        ('预测 rollout', 'oke_cv', 'oke_tr'),
                        ('层级 cv', 'okp_cv', 'oke_cv'),
                        ('层级 true', 'okp_tr', 'oke_tr')]:
        up, dn = flips(a, b)
        print(f'  {label:14s}: 拒→过 {up:5d} | 过→拒 {dn:5d} (净 {up-dn:+d})')

    print(f'\n===== 选择一致性（与 G0 不同）=====')
    for name, _ in POL[1:]:
        diff = sum(1 for sid in by if S[name][sid]['rank'] != base[sid]['rank'])
        print(f'  {name:8s}: {diff}/{n} ({100*diff/n:.1f}%)')

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=160)
    names = [p[0] for p in POL]
    x = np.arange(len(names))
    coll = [summ[p]['coll'] for p in names]
    off = [summ[p]['off'] for p in names]
    ax.bar(x - 0.18, coll, 0.34, label='collision', color='#e53935')
    ax.bar(x + 0.18, off, 0.34, label='off-road', color='#fb8c00')
    for i in range(len(names)):
        ax.text(x[i] - 0.18, coll[i] + 0.8, str(coll[i]), ha='center', fontsize=8)
        ax.text(x[i] + 0.18, off[i] + 0.8, str(off[i]), ha='center', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8, rotation=12)
    ax.set_ylabel(f'scenes (n={n})')
    ax.set_title('Round 2: aligned criteria, common rollout start')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(f'{args.out}/fig_grid2_fixed.png')
    plt.close(fig)
    print('\n图已保存', args.out)


if __name__ == '__main__':
    main()
