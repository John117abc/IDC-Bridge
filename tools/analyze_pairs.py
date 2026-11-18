"""配对分析：策略两两之间的 救回 / 新增 + 选择一致性。"""
import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', default='/workspace/data/grid2/grid2.npz')
    args = ap.parse_args()
    z = np.load(args.npz, allow_pickle=True)
    rows = json.loads(str(z['rows']))
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

    POL = [('G0', None), ('G1cv', 'okp_cv'), ('G1true', 'okp_tr'),
           ('G2cv', 'oke_cv'), ('G2true', 'oke_tr')]

    def first_ok(srows, key):
        for r in srows:
            if r[key]:
                return r
        return srows[0]

    S = {n: {sid: (v[0] if k is None else first_ok(v, k))
             for sid, v in by.items()} for n, k in POL}

    def pair(a, b):
        same = 0
        rescue = harm = both_bad = both_good = 0
        for sid in by:
            ra, rb = S[a][sid], S[b][sid]
            if ra['rank'] == rb['rank']:
                same += 1
            if ra['bad'] and not rb['bad']:
                rescue += 1          # b 救回
            elif (not ra['bad']) and rb['bad']:
                harm += 1            # b 新增
            elif ra['bad'] and rb['bad']:
                both_bad += 1
            else:
                both_good += 1
        return same, rescue, harm, both_bad, both_good

    print(f'{"对照":22s} {"选中相同":>8s} {"B救回":>6s} {"B新增":>6s} '
          f'{"都失败":>6s} {"都好":>6s}')
    pairs = [('G1cv', 'G2cv'), ('G1true', 'G2true'),
             ('G1cv', 'G1true'), ('G2cv', 'G2true'),
             ('G0', 'G1cv'), ('G0', 'G2cv')]
    n = len(by)
    for a, b in pairs:
        same, res, harm, bb, bg = pair(a, b)
        print(f'{a+" -> "+b:22s} {100*same/n:7.1f}% {res:6d} {harm:6d} '
              f'{bb:6d} {bg:6d}')

    print('\n（"B救回"=A失败而B成功；"B新增"=A成功而B失败）')


if __name__ == '__main__':
    main()
