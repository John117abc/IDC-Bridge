"""修 eval 的两处：① 不重复采样 ② 规划窗之后接回原参考（blend_tail）。"""
import ast

ep = '/gpudrive/baselines/IDC-Bridge/src/scripts/eval/idc-eval.py'
src = open(ep).read()
n = 0

if 'sample_with_replacement=True' in src:
    src = src.replace('sample_with_replacement=True',
                      'sample_with_replacement=False', 1)
    n += 1

old = """        install_external(builder_, ego_idx, paths_, speeds_,
                         target_pos=tgt_,
                         last_valid_idx={w: 41 for w in paths_},
                         blend_tail=False, maps=_maps)   # 两种模式同一套参考布局"""
new = """        install_external(builder_, ego_idx, paths_, speeds_,
                         target_pos=tgt_,
                         last_valid_idx={w: 41 for w in paths_},
                         blend_tail=True, tail_idx=41, maps=_maps)"""
if 'blend_tail=False' in src:
    src = src.replace(old, new, 1)
    n += 1

open(ep, 'w').write(src)
ast.parse(open(ep).read())
print('PATCH OK 改动处数 =', n,
      ' 不重复采样 =', 'sample_with_replacement=False' in open(ep).read(),
      ' blend_tail=True =', 'blend_tail=True' in open(ep).read())
