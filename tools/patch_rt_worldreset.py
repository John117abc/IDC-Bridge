"""世界被替换时重置规划器缓存（否则 buf/last 是旧世界的，规划直接跑飞）。"""
import ast

f = '/gpudrive/baselines/IDC-Bridge/src/scripts/train/idc-train.py'
s = open(f, encoding='utf-8-sig').read()

# 1) 模块级快照
old_cache = "_PLANNER_CACHE = {'rt': None, 'install': None}"
new_cache = "_PLANNER_CACHE = {'rt': None, 'install': None, 'worlds': None}"
if old_cache in s:
    s = s.replace(old_cache, new_cache, 1)

# 2) 步循环里检测 world 集合变化
anchor = """                try:
                    _rt = _PLANNER_CACHE['rt']"""
add = """                # 世界被替换（或本步结束）时重置规划器，避免用旧世界数据出规划
                _cur_worlds = tuple(sorted(int(i) for i in env.data_loader.indices))
                if _PLANNER_CACHE['worlds'] != _cur_worlds:
                    if _PLANNER_CACHE['worlds'] is not None:
                        _PLANNER_CACHE['rt'].reset()
                        logger.info('[RT] 世界已替换 → 重置规划器缓存')
                    _PLANNER_CACHE['worlds'] = _cur_worlds
                try:
                    _rt = _PLANNER_CACHE['rt']"""
if "世界已替换 → 重置规划器缓存" not in s:
    assert anchor in s, '找不到 RT try 锚点'
    s = s.replace(anchor, add, 1)
    open(f, 'w', encoding='utf-8').write(s.lstrip('\ufeff'))
    ast.parse(open(f, encoding='utf-8').read())
    print('PATCH OK')
else:
    print('已存在')
