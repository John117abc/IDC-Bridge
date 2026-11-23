"""修回 @staticmethod：把装饰器还给 _densify。"""
import ast

pr = '/gpudrive/baselines/IDC-Bridge/src/env/planner_runtime.py'
src = open(pr).read()

bad = """        return v[:n][None].astype(np.float32)

    @staticmethod
    def _densify"""
if bad in src:
    print('装饰器位置正确，无需修')
else:
    # 现状：@staticmethod 在 _speeds_from_path 上，_densify 前没有
    src = src.replace("""    @staticmethod
    def _speeds_from_path""", """    def _speeds_from_path""", 1)
    src = src.replace("""    def _densify""", """    @staticmethod
    def _densify""", 1)
    open(pr, 'w').write(src)
    ast.parse(open(pr).read())
    print('FIXED')

s = open(pr).read()
i = s.find('def _speeds_from_path')
j = s.find('def _densify')
print('--- _speeds_from_path 前 2 行 ---')
print('\n'.join(s[:i].splitlines()[-2:]))
print('--- _densify 前 2 行 ---')
print('\n'.join(s[:j].splitlines()[-2:]))
