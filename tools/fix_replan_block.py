"""修复 maybe_replan 被插坏的区域。"""
import ast

pr = '/gpudrive/baselines/IDC-Bridge/src/env/planner_runtime.py'
s = open(pr, encoding='utf-8-sig').read()

start = s.find("        if step % self.replan_every != 0:\n            # ---- 诊断")
end = s.find("        # 抓一份 env 自己的 91 点参考")
assert start > 0 and end > start, '找不到损坏区域'

fixed = """        if step % self.replan_every != 0:
            return None
"""
s = s[:start] + fixed + s[end:]
open(pr, 'w', encoding='utf-8').write(s.lstrip('\ufeff'))
ast.parse(open(pr, encoding='utf-8').read())
print('REPAIRED')
i = s.find('def maybe_replan')
print('\n'.join(s[i:].splitlines()[:4]))
