"""按文件路径加载模型/归一化模块，绕开 src/models/__init__.py 的连带导入（依赖 gpudrive）。"""
import ast

f = '/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/tools/idc_carla.py'
s = open(f, encoding='utf-8-sig').read()

old = """from models.continuous_actor_critic import TransformerActor  # noqa: E402
from utils.running_norm import RunningNorm                    # noqa: E402"""
new = """import importlib.util as _ilu


def _load_module(name, relpath):
    \"\"\"按文件路径加载单个模块（避免经包 __init__ 连带导入 gpudrive 依赖）。\"\"\"
    for base in _SRC_CANDIDATES:
        if not base:
            continue
        p = os.path.join(base, relpath)
        if os.path.isfile(p):
            spec = _ilu.spec_from_file_location(name, p)
            mod = _ilu.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise ImportError(f'找不到 {relpath}')


_cac = _load_module('_idc_cac', os.path.join('models', 'continuous_actor_critic.py'))
_rn = _load_module('_idc_running_norm', os.path.join('utils', 'running_norm.py'))
TransformerActor = _cac.TransformerActor
RunningNorm = _rn.RunningNorm"""
assert old in s, '找不到 import 块'
s = s.replace(old, new, 1)
open(f, 'w', encoding='utf-8').write(s.lstrip('\ufeff'))
ast.parse(open(f, encoding='utf-8').read())
print('PATCH OK')
