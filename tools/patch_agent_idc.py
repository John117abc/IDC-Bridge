"""把 IDC 控制器接进 plant_agent_vec.py（VEC_CTRL=idc 启用，默认仍纯追踪）。"""
import ast

f = '/gpudrive/baselines/IDC-Bridge/tools/plant_agent_vec.py'
s = open(f, encoding='utf-8-sig').read()
n = 0

# 1) 开关
if "VEC_CTRL = os.environ.get('VEC_CTRL'" not in s:
    anchor = "ROUTE_ONLY = os.environ.get('VEC_ROUTE_ONLY', '0') == '1'"
    s = s.replace(anchor, anchor + "\nVEC_CTRL = os.environ.get('VEC_CTRL', 'pp')  # pp|idc", 1)
    n += 1

# 2) 属性占位
if 'self._idc = None' not in s:
    anchor = "        self._atrack = {}"
    s = s.replace(anchor, anchor + "\n        self._idc = None", 1)
    n += 1

# 3) run_step：惰性创建控制器
if 'IDCController(' not in s:
    anchor = "        pose = self._pose()"
    add = """        pose = self._pose()
        if VEC_CTRL == 'idc' and self._idc is None:
            from plant_agent_idc import IDCController
            _town = (self._vehicle.get_world().get_map().name
                     .split('/')[-1].replace('_Opt', ''))
            self._idc = IDCController(
                os.environ.get('VEC_IDC_CKPT', VEC_CKPT),
                self._vehicle.get_world(), self._vehicle,
                os.path.join(MAPS, f'map_{_town}.npz'), infer_every=2)"""
    assert anchor in s, '找不到 pose = self._pose()'
    s = s.replace(anchor, add, 1)
    n += 1

# 4) 控制分支：把 _control 调用换成 IDC
old_ctrl = "        steer, thr, brk = self._control(wp41, pose[3])"
new_ctrl = """        if VEC_CTRL == 'idc' and self._idc is not None:
            steer, thr, brk = self._idc.control(pose, wp, pose[3])
        else:
            steer, thr, brk = self._control(wp41, pose[3])"""
if old_ctrl in s and "self._idc.control(pose, wp, pose[3])" not in s:
    s = s.replace(old_ctrl, new_ctrl, 1)
    n += 1

open(f, 'w', encoding='utf-8').write(s.lstrip('\ufeff'))
ast.parse(open(f, encoding='utf-8').read())
print('PATCH OK 处数 =', n,
      '| ctrl =', 'VEC_CTRL' in open(f, encoding='utf-8').read(),
      '| idc_call =', 'self._idc.control' in open(f, encoding='utf-8').read())
