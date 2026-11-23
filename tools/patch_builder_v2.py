"""给 idc_state_builder 加 v2 分支（IDC_STATE_V2>=1 时输出 84/102 维）。

v2 状态由 tools/idc_state_v2.build_state_v2 生成；本补丁只负责：
  · 取自车位姿/速度（abs/rel tensor）
  · 取 8 辆最近周车（partner tensor，已是 ego 系相对量 → 用位姿转回世界再交给 v2）
  · 取参考轨迹（candidate_paths[w][a][pid]，补 heading/speed/kappa）
"""
import ast

f = '/gpudrive/baselines/IDC-Bridge/src/env/idc_state_builder.py'
src = open(f, encoding='utf-8-sig').read()

# ---------- 1) 主入口加 v2 短路 ----------
old = """    def get_idc_observations_batch(self, ego_indices: list,
                                    num_other_vehicles: int = 8,
                                    path_indices: list = None,
                                    _abs_np=None, _rel_np=None,
                                    _partner_np=None) -> list:"""
new = """    def get_idc_observations_batch(self, ego_indices: list,
                                    num_other_vehicles: int = 8,
                                    path_indices: list = None,
                                    _abs_np=None, _rel_np=None,
                                    _partner_np=None) -> list:
        import os as _os
        if int(_os.environ.get('IDC_STATE_V2', '0')) >= 1:
            return self.get_idc_observations_v2(
                ego_indices, num_other_vehicles=num_other_vehicles,
                path_indices=path_indices, _abs_np=_abs_np,
                _rel_np=_rel_np, _partner_np=_partner_np)"""
assert old in src, '找不到主入口锚点'
src = src.replace(old, new, 1)

# ---------- 2) 追加 v2 实现（放在类末尾前）----------
anchor = "    def get_road_dist_batch(self, world_indices, temporal_indices,"
impl = '''    def get_idc_observations_v2(self, ego_indices, num_other_vehicles=8,
                                path_indices=None, _abs_np=None,
                                _rel_np=None, _partner_np=None):
        """v2 状态（84 维，含 6 模态时 102 维）。"""
        import os
        import sys
        import numpy as _np
        _tp = os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))), 'tools')
        if _tp not in sys.path:
            sys.path.insert(0, _tp)
        import idc_state_v2 as V2

        abs_np = (_abs_np if _abs_np is not None else
                  self.sim.absolute_self_observation_tensor().to_torch().cpu().numpy())
        rel_np = (_rel_np if _rel_np is not None else
                  self.sim.self_observation_tensor().to_torch().cpu().numpy())
        partner_np = (_partner_np if _partner_np is not None else
                      self.sim.partner_observations_tensor().to_torch().cpu().numpy())

        states = []
        for w in range(self.num_worlds):
            a = int(ego_indices[w])
            x = float(abs_np[w, a, 0]); y = float(abs_np[w, a, 1])
            h = float(abs_np[w, a, 7]); v = float(rel_np[w, a, 0])
            veh_len = float(abs_np[w, a, 10]) if abs_np.shape[-1] > 10 else 5.0
            if not (1.0 < veh_len < 15.0):
                veh_len = 5.286

            # 周车：partner 是 ego 系相对量 → 换回世界坐标
            nbrs = []
            c, s_ = _np.cos(h), _np.sin(h)
            for row in partner_np[w, a]:
                pv, rx, ry, rh = (float(row[0]), float(row[1]),
                                  float(row[2]), float(row[3]))
                if pv == 0.0 and rx == 0.0 and ry == 0.0:
                    continue
                if not _np.isfinite([pv, rx, ry, rh]).all():
                    continue
                nbrs.append(dict(x=x + rx * c - ry * s_, y=y + rx * s_ + ry * c,
                                 h=h + rh, v=pv))
            nbrs.sort(key=lambda q: (q['x'] - x) ** 2 + (q['y'] - y) ** 2)

            pid = path_indices[w] if path_indices is not None else 0
            path = self.candidate_paths[w][a][pid]
            st = V2.build_state_v2(
                dict(v=v, v_prev=v, omega=0.0, a_lat=0.0, steer_prev=0.0,
                     acc_prev=0.0, pose=(x, y, h)),
                nbrs, path, replan_phase=(self.step_counter[w] % 5) / 5.0,
                veh_len=veh_len, modes=None)
            states.append(st)
        return states

'''
assert anchor in src, '找不到 get_road_dist_batch 锚点'
src = src.replace(anchor, impl + anchor, 1)

open(f, 'w', encoding='utf-8').write(src.lstrip('\\ufeff'))
ast.parse(open(f, encoding='utf-8').read())
print('PATCH OK  v2 分支 =', 'get_idc_observations_v2' in open(f, encoding='utf-8').read())
