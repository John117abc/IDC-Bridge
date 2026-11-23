"""v2 模式下跳过 RunningNorm 更新（维度不一致，且 v2 用固定尺度）。"""
import ast

af = '/gpudrive/baselines/IDC-Bridge/src/agents/idc_agent.py'
s = open(af, encoding='utf-8-sig').read()

old = """            self.input_norm.update(enc_window[:, -1, :])"""
new = """            if getattr(self, 'state_ver', 0) == 0:
                self.input_norm.update(enc_window[:, -1, :])"""
assert old in s, '找不到 input_norm.update'
s = s.replace(old, new, 1)
open(af, 'w', encoding='utf-8').write(s.lstrip('\ufeff'))
ast.parse(open(af, encoding='utf-8').read())
print('PATCH OK')
