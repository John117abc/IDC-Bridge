"""把 planner 相关 CLI 参数加进 cli_overrides，否则 config 里拿不到。"""
import ast

f = '/gpudrive/baselines/IDC-Bridge/src/scripts/train/idc-train.py'
s = open(f, encoding='utf-8-sig').read()

old = "        'state_mode': args.state_mode,"
new = """        'state_mode': args.state_mode,
        'planner_runtime': args.planner_runtime,
        'planner_replan_every': args.planner_replan_every,
        'planner_k': args.planner_k,"""
assert old in s, '找不到 cli_overrides 锚点'
if "'planner_runtime': args.planner_runtime" not in s:
    s = s.replace(old, new, 1)
    open(f, 'w', encoding='utf-8').write(s.lstrip('\ufeff'))
    ast.parse(open(f, encoding='utf-8').read())
    print('PATCH OK')
else:
    print('已存在')
