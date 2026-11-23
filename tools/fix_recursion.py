"""去掉 _get_idc_obs_v1 内部的 v2 短路（避免自递归）。"""
import ast

bf = '/gpudrive/baselines/IDC-Bridge/src/env/idc_state_builder.py'
src = open(bf, encoding='utf-8-sig').read()

bad = """    def _get_idc_obs_v1(self, ego_indices: list,
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
good = """    def _get_idc_obs_v1(self, ego_indices: list,
                       num_other_vehicles: int = 8,
                       path_indices: list = None,
                       _abs_np=None, _rel_np=None,
                       _partner_np=None) -> list:"""
if bad in src:
    src = src.replace(bad, good, 1)
    print('已移除自递归短路')
else:
    print('未找到该模式，检查是否已修')

open(bf, 'w', encoding='utf-8').write(src.lstrip('\ufeff'))
ast.parse(open(bf, encoding='utf-8').read())
print('OK  剩余 v2 短路数 =', src.count("if int(_os.environ.get('IDC_STATE_V2'"))
