"""修正 v2 的 _encode 顺序：先 v1 的 64 维（与旧权重逐列对齐），再追加 19 维。"""
import ast

af = '/gpudrive/baselines/IDC-Bridge/src/agents/idc_agent.py'
s = open(af, encoding='utf-8-sig').read()

start = s.find("        if getattr(self, 'state_ver', 0) >= 1:\n            lead = s.shape[:-1]")
end = s.find("    def _norm(self, x: torch.Tensor)")
assert start > 0 and end > start, '找不到 v2 encode 块'

new_block = '''        if getattr(self, 'state_ver', 0) >= 1:
            lead = s.shape[:-1]
            x, y, phi = s[..., 0], s[..., 1], s[..., 4]
            z = torch.zeros_like(x)
            # --- 前 64 维：与 v1 编码**逐列一致**（保证旧权重可直接热启动）---
            ego4 = torch.stack([z, z, s[..., 2], z], dim=-1)          # 4
            o = s[..., 6:38].reshape(*lead, 8, 4)
            c, sn = torch.cos(phi), torch.sin(phi)
            dx = o[..., 0] - x.unsqueeze(-1)
            dy = o[..., 1] - y.unsqueeze(-1)
            rx = dx * c.unsqueeze(-1) + dy * sn.unsqueeze(-1)
            ry = -dx * sn.unsqueeze(-1) + dy * c.unsqueeze(-1)
            dphi = o[..., 2] - phi.unsqueeze(-1)
            dphi = torch.atan2(torch.sin(dphi), torch.cos(dphi))
            others = torch.stack([rx, ry, dphi, o[..., 3]], dim=-1).reshape(
                *lead, self.DIM_OTHERS)                                # 32
            rest = s[..., 38:65]                                       # 27
            veil = s[..., 84:85]                                       # 1
            base64 = torch.cat([ego4, others, rest, veil], dim=-1)      # 64
            # --- 后 19 维：追加特征（ego2 + others16 + replan1）---
            ego_ex = s[..., 65:67]
            oth_ex = s[..., 67:83]
            replan = s[..., 83:84]
            return torch.cat([base64, ego_ex, oth_ex, replan], dim=-1)  # 83

'''
s = s[:start] + new_block + s[end:]
open(af, 'w', encoding='utf-8').write(s.lstrip('\ufeff'))
ast.parse(open(af, encoding='utf-8').read())
print('ENCODE v2 顺序修正 OK')
