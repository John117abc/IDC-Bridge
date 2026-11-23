"""v2 模式下跳过 optimizer 状态加载（参数维度变了，旧状态不兼容）。"""
import ast

f = '/gpudrive/baselines/IDC-Bridge/src/utils/checkpoint.py'
s = open(f, encoding='utf-8-sig').read()

old = "    saved_optim_state = checkpoint.get('optimizer_state_dict')\n    if optimizer is not None and saved_optim_state is not None:"
new = """    import os as _os
    saved_optim_state = checkpoint.get('optimizer_state_dict')
    if int(_os.environ.get('IDC_STATE_V2', '0')) >= 1:
        print('[ckpt] IDC_STATE_V2>=1：跳过 optimizer 状态（维度已变）', flush=True)
        saved_optim_state = None
    if optimizer is not None and saved_optim_state is not None:"""
assert old in s, '找不到 optimizer 加载处'
s = s.replace(old, new, 1)
open(f, 'w', encoding='utf-8').write(s.lstrip('\ufeff'))
ast.parse(open(f, encoding='utf-8').read())
print('PATCH OK')
