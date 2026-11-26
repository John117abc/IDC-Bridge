import re

target = '/mnt/2T_HDD/plant2_vector/closedloop/fast_082540/run.log'
print('analyze:', target)
rows = []
pat = re.compile(r'\[PID\] v=([0-9.]+) v_tgt=([0-9.]+) steer=([+-]?[0-9.]+) '
                 r'thr=([0-9.]+) brk=([0-9.]+) off=([+-]?[0-9.]+)')
for line in open(target, encoding='utf-8', errors='replace'):
    m = pat.search(line)
    if m:
        rows.append(tuple(float(x) for x in m.groups()))

n = len(rows)
print('PID 样本', n)
v = [r[0] for r in rows]
vt = [r[1] for r in rows]
thr = [r[3] for r in rows]
brk = [r[4] for r in rows]

print(f'速度均值 {sum(v)/n:.2f}  最大 {max(v):.2f}')
print(f'目标速度均值 {sum(vt)/n:.2f}  最大 {max(vt):.2f}')
print(f'刹车>0.5: {sum(1 for b in brk if b>0.5)} ({100*sum(1 for b in brk if b>0.5)/n:.1f}%)')
print(f'满刹>=0.99: {sum(1 for b in brk if b>=0.99)} ({100*sum(1 for b in brk if b>=0.99)/n:.1f}%)')
print(f'油门>0.5: {100*sum(1 for t in thr if t>0.5)/n:.1f}%')
dv = [abs(vt[i]-vt[i-1]) for i in range(1, n)]
print(f'|Δv_tgt| 均值 {sum(dv)/len(dv):.2f} 最大 {max(dv):.2f}')
alt = sum(1 for i in range(1, n)
          if (brk[i] > 0.5 and thr[i-1] > 0.5) or (thr[i] > 0.5 and brk[i-1] > 0.5))
print(f'刹车/油门交替 {alt} ({100*alt/n:.1f}%)')

# 逐帧打印一段（看抖动）
print('\n连续 20 帧（v / v_tgt / thr / brk）:')
for r in rows[8:28]:
    print(f'  v={r[0]:5.2f} vt={r[1]:5.2f} thr={r[3]:4.2f} brk={r[4]:4.2f}')
