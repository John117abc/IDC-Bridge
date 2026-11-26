"""分析 100 分那次的纵向行为：急刹频率、速度波动"""
import glob
import re

import json, os
files = sorted(glob.glob('/mnt/2T_HDD/plant2_vector/closedloop/*/run.json'))
target = None
for j in files:
    try:
        d = json.load(open(j))
        rec = d['_checkpoint']['records']
        r0 = rec[0]
        if abs(r0['scores']['score_composed'] - 100.0) < 1e-6 and '24759' in r0.get('route_id', ''):
            target = os.path.join(os.path.dirname(j), 'run.log')
    except Exception:
        continue
print('analyze:', target)

rows = []
pat = re.compile(r'\[PID\] v=([0-9.]+) v_tgt=([0-9.]+) steer=([+-]?[0-9.]+) '
                 r'thr=([0-9.]+) brk=([0-9.]+) off=([+-]?[0-9.]+)')
for line in open(target, encoding='utf-8', errors='replace'):
    m = pat.search(line)
    if m:
        rows.append(tuple(float(x) for x in m.groups()))

n = len(rows)
if not n:
    print('无 PID 行')
    raise SystemExit

v = [r[0] for r in rows]
vt = [r[1] for r in rows]
thr = [r[3] for r in rows]
brk = [r[4] for r in rows]

hard = sum(1 for b in brk if b > 0.5)
full = sum(1 for b in brk if b >= 0.99)
print(f'样本 {n}')
print(f'速度: 均值 {sum(v)/n:.2f} 最大 {max(v):.2f} 最小 {min(v):.2f}')
print(f'目标速度: 均值 {sum(vt)/n:.2f} 最大 {max(vt):.2f}')
print(f'刹车>0.5: {hard} ({100*hard/n:.1f}%)   满刹>=0.99: {full} ({100*full/n:.1f}%)')
print(f'油门>0.5: {sum(1 for t in thr if t > 0.5)} ({100*sum(1 for t in thr if t>0.5)/n:.1f}%)')

dv = [abs(vt[i] - vt[i-1]) for i in range(1, n)]
print(f'|Δv_tgt| 相邻: 均值 {sum(dv)/len(dv):.2f} 最大 {max(dv):.2f} m/s')

alt = sum(1 for i in range(1, n)
          if (brk[i] > 0.5 and thr[i-1] > 0.5) or (thr[i] > 0.5 and brk[i-1] > 0.5))
print(f'刹车/油门交替: {alt} ({100*alt/n:.1f}%)')
