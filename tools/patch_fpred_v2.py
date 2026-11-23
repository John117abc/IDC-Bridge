"""f_pred_batch 在 v2 下保留追加的 19 维（单步近似：原样带过去）。"""
import ast

af = '/gpudrive/baselines/IDC-Bridge/src/agents/idc_agent.py'
s = open(af, encoding='utf-8-sig').read()

old = """        # 组合并返回最终状态
        return torch.cat([ego_next_formatted, others_next.view(len(states), -1),
                          validity_tensors, ref_error_tensors,
                          temporal_next.unsqueeze(-1).float(), veh_len], dim=-1)"""
new = """        # 组合并返回最终状态
        if getattr(self, 'state_ver', 0) >= 1:
            _base = (self.DIM_EGO + self.DIM_OTHERS + self.DIM_VALIDITY
                     + self.DIM_REF_ERROR + self.DIM_TEMPORAL)
            _extra = states[:, _base:_base + self.DIM_EXTRA]   # 单步近似：原样带过去
            return torch.cat([ego_next_formatted, others_next.view(len(states), -1),
                              validity_tensors, ref_error_tensors,
                              temporal_next.unsqueeze(-1).float(),
                              _extra, veh_len], dim=-1)
        return torch.cat([ego_next_formatted, others_next.view(len(states), -1),
                          validity_tensors, ref_error_tensors,
                          temporal_next.unsqueeze(-1).float(), veh_len], dim=-1)"""
assert old in s, '找不到 f_pred 返回处'
s = s.replace(old, new, 1)
open(af, 'w', encoding='utf-8').write(s.lstrip('\ufeff'))
ast.parse(open(af, encoding='utf-8').read())
print('PATCH OK')
