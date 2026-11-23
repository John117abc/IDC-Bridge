"""IDC 支持 v2 状态接口（84/102 维），用环境变量 IDC_STATE_V2=1 开启。

- 维度：ego 8 / others 48 / validity 8 / ref_error 18 / temporal 2 / veh 0 = 84
- `_encode`：v2 状态本身已在自车系且已按固定尺度归一化 → 恒等
- 归一化：关闭 RunningNorm（obs_norm=False），避免被旧统计污染
- 阶段 C（102 维）：由 IDC_STATE_V2=2 开启（多 6×3 模态块）
"""
import ast

f = '/gpudrive/baselines/IDC-Bridge/src/agents/idc_agent.py'
src = open(f).read()

# ---------- 1) 维度：原 6 行改成按环境变量分支 ----------
old_dims = """        self.DIM_EGO = 6
        self.DIM_OTHERS = 32
        self.DIM_VALIDITY = 8
        self.DIM_REF_ERROR = 18           # [dp, dphi, dv, lat+dphi+road+spd ×3(t+5/10/15) + curvahead+spdahead+rdahead]
        self.DIM_TEMPORAL = 1
        self.DIM_VEH = 1                  # 该车的 size.length（GPUDrive 拿它当轴距，逐车不同）"""
new_dims = """        import os as _os
        # 0=旧 66 维；1=v2 84 维（单参考）；2=v2 102 维（含 6 模态）
        self.state_ver = int(_os.environ.get('IDC_STATE_V2', '0'))
        if self.state_ver >= 1:
            self.DIM_EGO = 8              # v, v_prev, omega, a_lat, steer_prev, acc_prev, len, wid
            self.DIM_OTHERS = 48          # 8 辆 × (rel_x, rel_y, dphi, v, ttc, in_corridor)
            self.DIM_VALIDITY = 8
            self.DIM_REF_ERROR = 18       # dp,dphi,dv + 3×[lat,dphi,v_ref,kappa] + [lat,dphi,v_ref]
            self.DIM_TEMPORAL = 2         # 重规划相位, v/v_ref
            self.DIM_VEH = 0              # 已并入 ego
            self.DIM_MODE = 18 if self.state_ver >= 2 else 0
        else:
            self.DIM_EGO = 6
            self.DIM_OTHERS = 32
            self.DIM_VALIDITY = 8
            self.DIM_REF_ERROR = 18
            self.DIM_TEMPORAL = 1
            self.DIM_VEH = 1
            self.DIM_MODE = 0"""
assert old_dims in src, '找不到维度定义块'
src = src.replace(old_dims, new_dims, 1)

# ---------- 2) TOTAL_STATE_DIM 加 DIM_MODE ----------
old_tot = """        self.TOTAL_STATE_DIM = (self.DIM_EGO + self.DIM_OTHERS + self.DIM_VALIDITY
                                + self.DIM_REF_ERROR + self.DIM_TEMPORAL + self.DIM_VEH)"""
new_tot = """        self.TOTAL_STATE_DIM = (self.DIM_EGO + self.DIM_OTHERS + self.DIM_VALIDITY
                                + self.DIM_REF_ERROR + self.DIM_TEMPORAL + self.DIM_VEH
                                + getattr(self, 'DIM_MODE', 0))"""
assert old_tot in src
src = src.replace(old_tot, new_tot, 1)

# ---------- 3) ENCODED_EGO / ENCODED_DIM ----------
old_enc = """        self.ENCODED_EGO = 4"""
new_enc = """        self.ENCODED_EGO = self.DIM_EGO if getattr(self, 'state_ver', 0) >= 1 else 4"""
assert old_enc in src
src = src.replace(old_enc, new_enc, 1)

old_encdim = """        self.ENCODED_DIM = (self.ENCODED_EGO + self.DIM_OTHERS + self.DIM_VALIDITY
                            + self.DIM_REF_ERROR + self.DIM_TEMPORAL + self.DIM_VEH)"""
new_encdim = """        self.ENCODED_DIM = (self.ENCODED_EGO + self.DIM_OTHERS + self.DIM_VALIDITY
                            + self.DIM_REF_ERROR + self.DIM_TEMPORAL + self.DIM_VEH) \\
            if getattr(self, 'state_ver', 0) >= 1 else \\
            (self.ENCODED_EGO + self.DIM_OTHERS + self.DIM_VALIDITY
             + self.DIM_REF_ERROR + self.DIM_TEMPORAL + self.DIM_VEH)"""
assert old_encdim in src
src = src.replace(old_encdim, new_encdim, 1)

# ---------- 4) _encode：v2 恒等；_norm：v2 关闭 ----------
old_norm = """    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return self.input_norm(x) if self.obs_norm else x"""
new_norm = """    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        if getattr(self, 'state_ver', 0) >= 1:      # v2 状态已按固定尺度归一化
            return x
        return self.input_norm(x) if self.obs_norm else x"""
assert old_norm in src
src = src.replace(old_norm, new_norm, 1)

old_enc_fn = """        lead = s.shape[:-1]
        x, y, phi = s[..., 0], s[..., 1], s[..., 4]"""
new_enc_fn = """        if getattr(self, 'state_ver', 0) >= 1:
            return s                                # v2：已在自车系 + 已归一化
        lead = s.shape[:-1]
        x, y, phi = s[..., 0], s[..., 1], s[..., 4]"""
assert old_enc_fn in src
src = src.replace(old_enc_fn, new_enc_fn, 1)

open(f, 'w').write(src)
ast.parse(open(f).read())
s = open(f).read()
print('PATCH OK')
for k in ('state_ver', 'DIM_MODE', 'v2：已在自车系'):
    print('  ', k, '=', k in s)
