"""修 ENCODED_DIM：v2 要加上追加特征（19）与模态块。"""
import ast

af = '/gpudrive/baselines/IDC-Bridge/src/agents/idc_agent.py'
s = open(af, encoding='utf-8-sig').read()

old = """        self.ENCODED_DIM = (self.ENCODED_EGO + self.DIM_OTHERS + self.DIM_VALIDITY
                            + self.DIM_REF_ERROR + self.DIM_TEMPORAL + self.DIM_VEH) \\
            if getattr(self, 'state_ver', 0) >= 1 else \\
            (self.ENCODED_EGO + self.DIM_OTHERS + self.DIM_VALIDITY
             + self.DIM_REF_ERROR + self.DIM_TEMPORAL + self.DIM_VEH)"""
new = """        self.ENCODED_DIM = (self.ENCODED_EGO + self.DIM_OTHERS + self.DIM_VALIDITY
                            + self.DIM_REF_ERROR + self.DIM_TEMPORAL + self.DIM_VEH
                            + getattr(self, 'DIM_EXTRA', 0)
                            + getattr(self, 'DIM_MODE', 0))"""
if old in s:
    s = s.replace(old, new, 1)
    print('已替换 ENCODED_DIM')
elif 'self.ENCODED_DIM = (self.ENCODED_EGO + self.DIM_OTHERS + self.DIM_VALIDITY\n                            + self.DIM_REF_ERROR + self.DIM_TEMPORAL + self.DIM_VEH)' in s:
    s = s.replace("""        self.ENCODED_DIM = (self.ENCODED_EGO + self.DIM_OTHERS + self.DIM_VALIDITY
                            + self.DIM_REF_ERROR + self.DIM_TEMPORAL + self.DIM_VEH)""",
                  new, 1)
    print('已替换（短形式）')
else:
    print('未匹配，需人工检查')

open(af, 'w', encoding='utf-8').write(s.lstrip('\ufeff'))
ast.parse(open(af, encoding='utf-8').read())
import re
for line in open(af, encoding='utf-8').read().splitlines():
    if 'ENCODED_DIM =' in line or 'DIM_EXTRA' in line:
        print('  ', line.strip())
