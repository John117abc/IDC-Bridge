"""修速度通道：注入的 speeds 由规划器轨迹的隐含速度决定（不再用常数 20 m/s）。"""
import ast

pr = '/gpudrive/baselines/IDC-Bridge/src/env/planner_runtime.py'
src = open(pr).read()
n = 0

# 1) 加一个由路径推速度的方法（放在 _densify 之前）
anchor = "    def _densify"
helper = '''    def _speeds_from_path(self, path, n=None):
        """由轨迹点间距推隐含速度（dt=0.1 时 SPAN=1 点/0.1s）。"""
        P = np.asarray(path, np.float32)
        n = n or P.shape[0]
        if len(P) < 2:
            return np.full((1, n), 5.0, np.float32)
        d = np.linalg.norm(np.diff(P, axis=0), axis=1) / 0.1
        d = np.clip(d, 0.5, 25.0)
        v = np.r_[d, d[-1]]
        if len(v) < n:
            v = np.r_[v, np.full(n - len(v), v[-1])]
        return v[:n][None].astype(np.float32)

'''
if '_speeds_from_path' not in src:
    assert anchor in src, '找不到 _densify 锚点'
    src = src.replace(anchor, helper + anchor, 1)
    n += 1

# 2) 所有 speeds[w] = np.full(...) 改成由路径推
import re
pat_count = len(re.findall(r'speeds\[w\] = np\.full\(\(1, N_ENV\), self\.speed_ref, np\.float32\)', src))
src = re.sub(r'speeds\[w\] = np\.full\(\(1, N_ENV\), self\.speed_ref, np\.float32\)',
             'speeds[w] = self._speeds_from_path(paths[w][0], N_ENV)', src)
n += pat_count

open(pr, 'w').write(src)
ast.parse(open(pr).read())
print(f'PATCH OK  替换 speeds 处数 = {pat_count}, helper = {("_speeds_from_path" in open(pr).read())}')
