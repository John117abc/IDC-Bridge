"""checkpoint 加载改为容忍式：形状不匹配的参数跳过（用于 66→84 维迁移）。"""
import ast

f = '/gpudrive/baselines/IDC-Bridge/src/utils/checkpoint.py'
src = open(f, encoding='utf-8-sig').read()
n = 0

old = "            net.load_state_dict(saved_model_state[name])"
new = """            _sd = saved_model_state[name]
            try:
                net.load_state_dict(_sd)
            except RuntimeError as _e:
                # 维度升级（如 66→84 维）时首层形状不匹配：只跳过这些参数
                _own = net.state_dict()
                _ok, _skip = {}, []
                for _k, _v in _sd.items():
                    if _k in _own and _own[_k].shape == _v.shape:
                        _ok[_k] = _v
                    else:
                        _skip.append(_k)
                net.load_state_dict(_ok, strict=False)
                print(f'[ckpt] {name}: 跳过 {len(_skip)} 个形状不匹配参数 '
                      f'{_skip[:3]}（其余已加载）', flush=True)"""
if old in src:
    src = src.replace(old, new, 1)
    n += 1

open(f, 'w', encoding='utf-8').write(src.lstrip('\ufeff'))
ast.parse(open(f, encoding='utf-8').read())
print('PATCH OK 处数 =', n)
