"""补回 v1 的 _encode 分支（之前被覆盖掉了）。"""
import ast

af = '/gpudrive/baselines/IDC-Bridge/src/agents/idc_agent.py'
s = open(af, encoding='utf-8-sig').read()

anchor = """            return torch.cat([base64, ego_ex, oth_ex, replan], dim=-1)  # 83

    def _norm(self, x: torch.Tensor)"""
add = """            return torch.cat([base64, ego_ex, oth_ex, replan], dim=-1)  # 83

        # ---- v1（66 维）编码：与旧权重逐列一致 ----
        lead = s.shape[:-1]
        x, y, phi = s[..., 0], s[..., 1], s[..., 4]
        o = s[..., self.DIM_EGO:self.DIM_EGO + self.DIM_OTHERS].reshape(
            *lead, 8, 4)
        if getattr(self, 'state_mode', 'ego') == 'ego':
            c, sn = torch.cos(phi), torch.sin(phi)
            ex = torch.zeros_like(x)
            ey = torch.zeros_like(y)
            ephi = torch.zeros_like(phi)
            dx = o[..., 0] - x.unsqueeze(-1)
            dy = o[..., 1] - y.unsqueeze(-1)
            rx = dx * c.unsqueeze(-1) + dy * sn.unsqueeze(-1)
            ry = -dx * sn.unsqueeze(-1) + dy * c.unsqueeze(-1)
            dphi = o[..., 2] - phi.unsqueeze(-1)
            dphi = torch.atan2(torch.sin(dphi), torch.cos(dphi))
        else:
            ex, ey, ephi = x, y, phi
            rx, ry, dphi = o[..., 0], o[..., 1], o[..., 2]
        ego_enc = torch.stack([ex, ey, s[..., 2], ephi], dim=-1)
        others_enc = torch.stack([rx, ry, dphi, o[..., 3]], dim=-1).reshape(
            *lead, self.DIM_OTHERS)
        rest = s[..., self.DIM_EGO + self.DIM_OTHERS:]
        return torch.cat([ego_enc, others_enc, rest], dim=-1)

    def _norm(self, x: torch.Tensor)"""
assert anchor in s, '找不到 v2 结尾锚点'
if 'v1（66 维）编码' not in s:
    s = s.replace(anchor, add, 1)
    open(af, 'w', encoding='utf-8').write(s.lstrip('\ufeff'))
    ast.parse(open(af, encoding='utf-8').read())
    print('PATCH OK')
else:
    print('已存在')
