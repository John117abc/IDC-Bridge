"""去掉 BOM 并校验 idc_agent.py 的 v2 补丁是否完整。"""
import ast

f = '/gpudrive/baselines/IDC-Bridge/src/agents/idc_agent.py'
src = open(f, encoding='utf-8-sig').read()
# 去掉可能残留的 BOM 字符
src = src.lstrip('\ufeff')
open(f, 'w', encoding='utf-8').write(src)
ast.parse(src)
s = open(f, encoding='utf-8').read()
print('BOM 清除完成，语法 OK')
for k in ('state_ver', 'DIM_MODE', 'v2：已在自车系', 'ENCODED_EGO = self.DIM_EGO'):
    print('  ', k, '=', k in s)
print('  TOTAL_STATE_DIM 行:', [l.strip() for l in s.splitlines()
                               if 'TOTAL_STATE_DIM' in l][:2])
