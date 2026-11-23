"""首层维度扩展时做“列拷贝”初始化：旧权重填前 N 列，新列置零。
这样策略在初始时行为与旧模型一致（新增特征暂被忽略），避免随机首层把车开飞。
"""
import ast

f = '/gpudrive/baselines/IDC-Bridge/src/utils/checkpoint.py'
s = open(f, encoding='utf-8-sig').read()

old = """                _own = net.state_dict()
                _ok, _skip = {}, []
                for _k, _v in _sd.items():
                    if _k in _own and _own[_k].shape == _v.shape:
                        _ok[_k] = _v
                    else:
                        _skip.append(_k)
                net.load_state_dict(_ok, strict=False)"""
new = """                _own = net.state_dict()
                _ok, _skip, _expand = {}, [], []
                for _k, _v in _sd.items():
                    if _k in _own and _own[_k].shape == _v.shape:
                        _ok[_k] = _v
                    elif (_k in _own and _v.dim() == 2
                          and _own[_k].dim() == 2
                          and _v.shape[0] == _own[_k].shape[0]
                          and _v.shape[1] < _own[_k].shape[1]):
                        # 输入维度扩展：旧列拷到前面，新列置零（热启动）
                        _new = torch.zeros_like(_own[_k])
                        _new[:, :_v.shape[1]] = _v
                        _ok[_k] = _new
                        _expand.append((_k, tuple(_v.shape), tuple(_own[_k].shape)))
                    else:
                        _skip.append(_k)
                net.load_state_dict(_ok, strict=False)
                if _expand:
                    print(f'[ckpt] {name}: 维度扩展热启动 {_expand}', flush=True)"""
assert old in s, '找不到容忍加载块'
s = s.replace(old, new, 1)
if 'import torch' not in s.split('\n\n')[0]:
    s = 'import torch\n' + s
open(f, 'w', encoding='utf-8').write(s.lstrip('\ufeff'))
ast.parse(open(f, encoding='utf-8').read())
print('PATCH OK')
