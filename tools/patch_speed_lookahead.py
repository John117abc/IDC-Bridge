"""统一参考速度取法：speed[t+5]（前瞻 0.5 s），曲率限速上限 20 m/s。

训练侧（GPUDrive）：
  · idc_state_builder.get_idc_obs_v1：rs 取 path['speed'][t+5]
  · get_ref_states_batch：speed 取 t_proj+5（快路径 + CPU 回退）
部署侧（CARLA）：
  · idc_state_carla.build_state：dv 用 path['speed'][t+5]
"""
import ast

LOOKAHEAD = 5

# ---------- 1) GPUDrive builder：v1 核心 ----------
bf = '/gpudrive/baselines/IDC-Bridge/src/env/idc_state_builder.py'
s = open(bf, encoding='utf-8-sig').read()
n1 = 0

old = """            rx, ry = float(path['pos'][t, 0]), float(path['pos'][t, 1])
            rh, rs = float(path['heading'][t]), float(path['speed'][t])"""
new = """            rx, ry = float(path['pos'][t, 0]), float(path['pos'][t, 1])
            rh = float(path['heading'][t])
            # 参考速度取 t+5（前瞻 0.5 s）：路径起点的期望速度常≈0，直接取 t 会让车"起步即死锁"
            _ts = int(min(t + 5, len(path['speed']) - 1))
            rs = float(path['speed'][_ts])"""
if old in s:
    s = s.replace(old, new, 1)
    n1 += 1

# ---------- 2) get_ref_states_batch：快路径 ----------
old2 = """            vals = self.ref_tensor[w_tensor, 0, p_tensor, t_proj]
            rx, ry, rs, rh, rd = vals[:, 0], vals[:, 1], vals[:, 2], vals[:, 3], vals[:, 4]"""
new2 = """            t_spd = torch.clamp(t_proj + 5, max=self.ref_tensor.shape[3] - 1)
            vals = self.ref_tensor[w_tensor, 0, p_tensor, t_proj]
            vals_spd = self.ref_tensor[w_tensor, 0, p_tensor, t_spd]
            rx, ry, rs, rh, rd = vals[:, 0], vals[:, 1], vals_spd[:, 2], vals[:, 3], vals[:, 4]"""
if old2 in s:
    s = s.replace(old2, new2, 1)
    n1 += 1

open(bf, 'w', encoding='utf-8').write(s.lstrip('\ufeff'))
ast.parse(open(bf, encoding='utf-8').read())
print('GPUDrive builder 改动数 =', n1)

# ---------- 3) CARLA 侧 ----------
cf = '/gpudrive/baselines/IDC-Bridge/tools/idc_state_carla.py'
s2 = open(cf, encoding='utf-8-sig').read()
old3 = """    ref = np.array([pos[t, 0], pos[t, 1], path['speed'][t], 0.0,
                    path['heading'][t], 0.0], np.float32)"""
new3 = """    _ts = int(min(t + 5, len(path['speed']) - 1))   # 前瞻 0.5 s 取参考速度
    ref = np.array([pos[t, 0], pos[t, 1], path['speed'][_ts], 0.0,
                    path['heading'][t], 0.0], np.float32)"""
assert old3 in s2, 'CARLA build_state 锚点没找到'
s2 = s2.replace(old3, new3, 1)
open(cf, 'w', encoding='utf-8').write(s2.lstrip('\ufeff'))
ast.parse(open(cf, encoding='utf-8').read())
print('CARLA 侧改动 OK')
