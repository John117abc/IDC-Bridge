"""从 Bench2Drive 路线 xml 里挑转向最大的 route（按路点几何算），生成子集 xml。"""
import sys
import xml.etree.ElementTree as ET

import numpy as np

SRC = sys.argv[1] if len(sys.argv) > 1 else \
    '/home/jiangchengxuan/PycharmProjects/research-safe-drive/plant2/Bench2Drive/leaderboard/data/bench2drive220.xml'
OUT = sys.argv[2] if len(sys.argv) > 2 else '/mnt/2T_HDD/plant2_vector/turn_routes.xml'
N = int(sys.argv[3]) if len(sys.argv) > 3 else 4

tree = ET.parse(SRC)
root = tree.getroot()
rows = []
for r in root.findall('route'):
    wps = r.find('waypoints')
    if wps is None:
        continue
    P = np.array([[float(p.get('x')), float(p.get('y'))]
                  for p in wps.findall('position')])
    if len(P) < 10:
        continue
    d = np.diff(P, axis=0)
    L = np.linalg.norm(d, axis=1)
    keep = L > 1e-3
    if keep.sum() < 5:
        continue
    th = np.unwrap(np.arctan2(d[keep, 1], d[keep, 0]))
    dh = float(np.abs(np.diff(th)).sum())
    rows.append((dh, float(L.sum()), r.get('id'), r.get('town'), r))

rows.sort(key=lambda x: -x[0])
print('候选路线数', len(rows))
print('%-8s %-8s %8s %8s' % ('id', 'town', '转角(rad)', '长度(m)'))
for dh, ln, rid, town, _ in rows[:10]:
    print('%-8s %-8s %8.2f %8.1f' % (rid, town, dh, ln))

sel = rows[:N]
new = ET.Element('routes')
for dh, ln, rid, town, r in sel:
    new.append(r)
ET.ElementTree(new).write(OUT)
print('已写', OUT, '包含', len(sel), '条:',
      [(rid, town, round(dh, 2)) for dh, ln, rid, town, _ in sel])
