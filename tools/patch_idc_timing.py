"""给 IDC 控制器加计时（每 100 次推理打印一次），并支持降低频率。"""
import ast

f = '/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/tools/plant_agent_idc.py'
s = open(f, encoding='utf-8-sig').read()

# 计时字段
if 'self._t_cost' not in s:
    s = s.replace("        self.prev_v = None\n        self.a_lat = 0.0",
                  "        self.prev_v = None\n        self.a_lat = 0.0\n"
                  "        self._t_cost = 0.0\n        self._n_infer = 0", 1)

# 计时包裹推理
old = """            delta, acc = self.pol.act(st)"""
new = """            import time as _tm
            _t0 = _tm.time()
            delta, acc = self.pol.act(st)
            self._t_cost += _tm.time() - _t0
            self._n_infer += 1
            if self._n_infer % 100 == 0:
                print(f'[IDC] 推理 {self._n_infer} 次, 平均 '
                      f'{self._t_cost / self._n_infer * 1000:.1f} ms/次, '
                      f'device={self.pol.dev}', flush=True)"""
assert old in s
s = s.replace(old, new, 1)
open(f, 'w', encoding='utf-8').write(s.lstrip('\ufeff'))
ast.parse(open(f, encoding='utf-8').read())
print('PATCH OK')
