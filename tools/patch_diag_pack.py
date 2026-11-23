"""加临时诊断：pack 分支里打印 seg 数与最近距离。"""
import ast

pr = '/gpudrive/baselines/IDC-Bridge/src/env/planner_runtime.py'
src = open(pr).read()

anchor = """                if pk is not None:
                    _t = min(10 + int(step), 90)"""
add = """                if pk is not None:
                    if self.debug and step == 0:
                        _sa = pk.seg_a
                        _dd = (float(np.linalg.norm(_sa - np.array([x, y])[None], axis=1).min())
                               if len(_sa) else -1.0)
                        print('[RT][pack] w=%d seg=%d agents=%d ego_pos=%d '
                              'nearest_seg_d=%.1f first_seg=%s'
                              % (w, len(_sa), len(pk.agents), len(pk.ego_pos),
                                 _dd, np.round(_sa[0], 1) if len(_sa) else None),
                              flush=True)
                    _t = min(10 + int(step), 90)"""
if '[RT][pack]' not in src:
    assert anchor in src, '找不到 pack 分支锚点'
    src = src.replace(anchor, add, 1)
    open(pr, 'w').write(src)
ast.parse(open(pr).read())
print('DIAG PATCH OK')
