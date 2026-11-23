"""从 bench2drive220 里挑几条不同城镇的路线，生成子集 xml。"""
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

SRC = '/home/jiangchengxuan/PycharmProjects/research-safe-drive/plant2/Bench2Drive/leaderboard/data/bench2drive220.xml'
OUT = '/mnt/2T_HDD/plant2_vector/other_towns.xml'
WANT = ['Town03', 'Town05', 'Town10HD', 'Town04']

tree = ET.parse(SRC)
root = tree.getroot()
by_town = defaultdict(list)
for r in root.findall('route'):
    by_town[r.get('town')].append(r)

print('可用城镇:', {k: len(v) for k, v in by_town.items()})

sel = []
for t in WANT:
    rs = by_town.get(t) or by_town.get(t.replace('HD', '')) or []
    if rs:
        sel.append(rs[0])
        print(f'选用 {t}: route id={rs[0].get("id")}')

new = ET.Element('routes')
for r in sel:
    new.append(r)
ET.ElementTree(new).write(OUT)
print('已写', OUT, len(sel), '条')
