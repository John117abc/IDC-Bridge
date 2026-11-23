"""补 v2 的 _encode：85 → 83（去绝对坐标、周车转 ego 系、保留追加特征）。"""
import ast

af = '/gpudrive/baselines/IDC-Bridge/src/agents/idc_agent.py'
s = open(af, encoding='utf-8-sig').read()

old = """        if getattr(self, 'state_ver', 0) >= 1:
            return s                                # v2：已在自车系 + 已归一化"""
new = """        if getattr(self, 'state_ver', 0) >= 1:
            lead = s.shape[:-1]
            x, y, phi = s[..., 0], s[..., 1], s[..., 4]
            z = torch.zeros_like(x)
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
            ego_ex = s[..., 65:67]                                    # 2
            oth_ex = s[..., 67:83]                                    # 16
            rest = s[..., 38:65]                                      # 27
            replan = s[..., 83:84]                                    # 1
            veil = s[..., 84:85]                                      # 1
            return torch.cat([ego4, ego_ex, others, oth_ex, rest,
                              replan, veil], dim=-1)     # 4+2+32+16+27+1+1 = 83"""
assert old in s, '找不到 v2 恒等返回'
s = s.replace(old, new, 1)
open(af, 'w', encoding='utf-8').write(s.lstrip('\ufeff'))
ast.parse(open(af, encoding='utf-8').read())
print('ENCODE OK')
