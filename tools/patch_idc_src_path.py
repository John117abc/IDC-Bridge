"""让 idc_carla.py 在 CARLA(宿主机) 环境也能找到 IDC-Bridge/src 的模型与归一化类。"""
import ast

f = '/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/tools/idc_carla.py'
s = open(f, encoding='utf-8-sig').read()

old = """for _p in (_REPO, os.path.join(_REPO, 'src'), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)"""
new = """_SRC_CANDIDATES = [os.path.join(_REPO, 'src'), _HERE,
                   os.environ.get('VEC_IDC_SRC', ''),
                   '/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/src']
for _p in _SRC_CANDIDATES:
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)"""
assert old in s, '找不到 sys.path 块'
s = s.replace(old, new, 1)
open(f, 'w', encoding='utf-8').write(s.lstrip('\ufeff'))
ast.parse(open(f, encoding='utf-8').read())
print('PATCH OK')
