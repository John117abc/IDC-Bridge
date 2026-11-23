"""v2 归一化修正：前 64 维沿用旧 RunningNorm（保证与旧模型逐位一致），
   新增 18 维用固定尺度（已在构造时归一化，直接透传）。"""
import ast

af = '/gpudrive/baselines/IDC-Bridge/src/agents/idc_agent.py'
s = open(af, encoding='utf-8-sig').read()

# 1) input_norm 维度：v2 恒为 64（只管 base 部分）
old = "        self.input_norm = RunningNorm(self.ENCODED_DIM)"
new = """        self.input_norm = RunningNorm(
            64 if getattr(self, 'state_ver', 0) >= 1 else self.ENCODED_DIM)"""
if old in s:
    s = s.replace(old, new, 1)

# 2) _norm：v2 拆成 base(64) 归一化 + extras 透传
old_norm = """    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        if getattr(self, 'state_ver', 0) >= 1:      # v2 状态已按固定尺度归一化
            return x
        return self.input_norm(x) if self.obs_norm else x"""
new_norm = """    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        if getattr(self, 'state_ver', 0) >= 1:
            # 前 64 维 = 旧模型口径（RunningNorm）；后 18 维已在构造时归一化
            base = x[..., :64]
            base = self.input_norm(base) if self.obs_norm else base
            return torch.cat([base, x[..., 64:]], dim=-1)
        return self.input_norm(x) if self.obs_norm else x"""
assert old_norm in s, '找不到 _norm'
s = s.replace(old_norm, new_norm, 1)

# 3) select_action 里允许 v2 更新 base 统计
old_upd = """            if getattr(self, 'state_ver', 0) == 0:
                self.input_norm.update(enc_window[:, -1, :])"""
new_upd = """            if getattr(self, 'state_ver', 0) == 0:
                self.input_norm.update(enc_window[:, -1, :])
            else:
                self.input_norm.update(enc_window[:, -1, :64])"""
if old_upd in s:
    s = s.replace(old_upd, new_upd, 1)

open(af, 'w', encoding='utf-8').write(s.lstrip('\ufeff'))
ast.parse(open(af, encoding='utf-8').read())
print('PATCH OK  input_norm=64:', 'RunningNorm(\n            64 if' in open(af, encoding='utf-8').read())
