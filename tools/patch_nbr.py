"""修 make_state84：把 dict 周车转成 idc_state_carla 期望的 list 形式。"""
import ast

f = '/gpudrive/baselines/IDC-Bridge/tools/idc_carla.py'
s = open(f, encoding='utf-8-sig').read()

old = """    core66, t = S.build_state(ego_pose_canon, speed, neighbors_canon, path,
                              ref_idx=ref_idx, veh_len=veh_len)"""
new = """    _nbr = [[float(q['x']), float(q['y']), float(q['h']), float(q['v'])]
            for q in neighbors_canon]
    core66, t = S.build_state(ego_pose_canon, speed, _nbr, path,
                              ref_idx=ref_idx, veh_len=veh_len)"""
assert old in s
s = s.replace(old, new, 1)
open(f, 'w', encoding='utf-8').write(s.lstrip('\ufeff'))
ast.parse(open(f, encoding='utf-8').read())
print('PATCH OK')
