"""把 builder 的 v2 分支改成「v1 核心 + 追加特征（85 维）」，并修 agent 的维度/编码。"""
import ast

# ---------------- 1) builder ----------------
bf = '/gpudrive/baselines/IDC-Bridge/src/env/idc_state_builder.py'
src = open(bf, encoding='utf-8-sig').read()

start = src.find('    def get_idc_observations_v2(')
end = src.find('    def get_road_dist_batch(')
assert start > 0 and end > start, '找不到 v2 方法范围'

new_impl = '''    def get_idc_observations_v2(self, ego_indices, num_other_vehicles=8,
                                path_indices=None, _abs_np=None,
                                _rel_np=None, _partner_np=None):
        """v2 = v1 核心（66，绝对坐标）+ 追加特征（19）= 85 维。"""
        import os
        import sys
        import numpy as _np
        _tp = os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))), 'tools')
        if _tp not in sys.path:
            sys.path.insert(0, _tp)
        import idc_state_v2 as V2

        core = self._get_idc_obs_v1(
            ego_indices, num_other_vehicles=num_other_vehicles,
            path_indices=path_indices, _abs_np=_abs_np,
            _rel_np=_rel_np, _partner_np=_partner_np)

        abs_np = (_abs_np if _abs_np is not None else
                  self.sim.absolute_self_observation_tensor().to_torch().cpu().numpy())
        rel_np = (_rel_np if _rel_np is not None else
                  self.sim.self_observation_tensor().to_torch().cpu().numpy())
        partner_np = (_partner_np if _partner_np is not None else
                      self.sim.partner_observations_tensor().to_torch().cpu().numpy())

        out = []
        for w, st in enumerate(core):
            a = int(ego_indices[w])
            x = float(abs_np[w, a, 0]); y = float(abs_np[w, a, 1])
            h = float(abs_np[w, a, 7]); v = float(rel_np[w, a, 0])
            c, s_ = _np.cos(h), _np.sin(h)
            nbrs = []
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
            out.append(V2.build_v2(st, (x, y, h), nbrs,
                                   step=self.step_counter[w], v_ego=v))
        return out

'''
src = src[:start] + new_impl + src[end:]

# 原入口改名为 _get_idc_obs_v1，并保留 v2 短路
src = src.replace("""    def get_idc_observations_batch(self, ego_indices: list,
                                    num_other_vehicles: int = 8,
                                    path_indices: list = None,
                                    _abs_np=None, _rel_np=None,
                                    _partner_np=None) -> list:""",
                  """    def _get_idc_obs_v1(self, ego_indices: list,
                       num_other_vehicles: int = 8,
                       path_indices: list = None,
                       _abs_np=None, _rel_np=None,
                       _partner_np=None) -> list:""", 1)

# 新增对外入口
anchor = "    def get_idc_observations_v2("
entry = '''    def get_idc_observations_batch(self, ego_indices: list,
                                    num_other_vehicles: int = 8,
                                    path_indices: list = None,
                                    _abs_np=None, _rel_np=None,
                                    _partner_np=None) -> list:
        import os as _os
        if int(_os.environ.get('IDC_STATE_V2', '0')) >= 1:
            return self.get_idc_observations_v2(
                ego_indices, num_other_vehicles=num_other_vehicles,
                path_indices=path_indices, _abs_np=_abs_np,
                _rel_np=_rel_np, _partner_np=_partner_np)
        return self._get_idc_obs_v1(
            ego_indices, num_other_vehicles=num_other_vehicles,
            path_indices=path_indices, _abs_np=_abs_np,
            _rel_np=_rel_np, _partner_np=_partner_np)

'''
src = src.replace(anchor, entry + anchor, 1)
open(bf, 'w', encoding='utf-8').write(src.lstrip('\ufeff'))
ast.parse(open(bf, encoding='utf-8').read())
print('BUILDER OK')

# ---------------- 2) agent 维度 ----------------
af = '/gpudrive/baselines/IDC-Bridge/src/agents/idc_agent.py'
s = open(af, encoding='utf-8-sig').read()
s = s.replace("""        if self.state_ver >= 1:
            self.DIM_EGO = 8              # v, v_prev, omega, a_lat, steer_prev, acc_prev, len, wid
            self.DIM_OTHERS = 48          # 8 辆 × (rel_x, rel_y, dphi, v, ttc, in_corridor)
            self.DIM_VALIDITY = 8
            self.DIM_REF_ERROR = 18       # dp,dphi,dv + 3×[lat,dphi,v_ref,kappa] + [lat,dphi,v_ref]
            self.DIM_TEMPORAL = 2         # 重规划相位, v/v_ref
            self.DIM_VEH = 0              # 已并入 ego
            self.DIM_MODE = 18 if self.state_ver >= 2 else 0""",
              """        if self.state_ver >= 1:
            # v2：v1 核心（ego6/others32/valid8/ref18/temp1/veh1 = 66）+ 追加 19
            self.DIM_EGO = 6
            self.DIM_OTHERS = 32
            self.DIM_VALIDITY = 8
            self.DIM_REF_ERROR = 18
            self.DIM_TEMPORAL = 1
            self.DIM_VEH = 1
            self.DIM_EXTRA = 19           # ego2 + others16 + replan1（插在 veh 之前）
            self.DIM_MODE = 0""", 1)
s = s.replace("""        self.TOTAL_STATE_DIM = (self.DIM_EGO + self.DIM_OTHERS + self.DIM_VALIDITY
                                + self.DIM_REF_ERROR + self.DIM_TEMPORAL + self.DIM_VEH
                                + getattr(self, 'DIM_MODE', 0))""",
              """        self.TOTAL_STATE_DIM = (self.DIM_EGO + self.DIM_OTHERS + self.DIM_VALIDITY
                                + self.DIM_REF_ERROR + self.DIM_TEMPORAL + self.DIM_VEH
                                + getattr(self, 'DIM_EXTRA', 0)
                                + getattr(self, 'DIM_MODE', 0))""", 1)
s = s.replace("        self.ENCODED_EGO = self.DIM_EGO if getattr(self, 'state_ver', 0) >= 1 else 4",
              "        self.ENCODED_EGO = 4", 1)
open(af, 'w', encoding='utf-8').write(s.lstrip('\ufeff'))
ast.parse(open(af, encoding='utf-8').read())
print('AGENT DIM OK  TOTAL =', 'DIM_EXTRA' in open(af, encoding='utf-8').read())
