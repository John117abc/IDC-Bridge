"""把 v2 精确改成 84 维 = v1 核心 66 + 追加 18（ego2 + others16），去掉 replan 1 维。

同步修改：
  tools/idc_state_v2.py         EXTRA_DIM 18，build_v2 不再拼 replan
  src/env/idc_state_builder.py  build_v2 调用去掉 step 参数
  src/agents/idc_agent.py       DIM_EXTRA=18；_encode 只追加 18 维
"""
import ast
import re

# ---------- 1) idc_state_v2.py ----------
p1 = '/gpudrive/baselines/IDC-Bridge/tools/idc_state_v2.py'
s1 = open(p1, encoding='utf-8-sig').read()
s1 = s1.replace("EXTRA_DIM = EXTRA_EGO + N_OTHER * EXTRA_OTHER + 1        # 19",
                "EXTRA_DIM = EXTRA_EGO + N_OTHER * EXTRA_OTHER            # 18")
s1 = s1.replace("V2_DIM = V1_DIM + EXTRA_DIM     # 85", "V2_DIM = V1_DIM + EXTRA_DIM     # 84")
s1 = s1.replace("""def build_v2(core66, pose, neighbors, a_lat=0.0, steer_prev=0.0,
             step=0, replan_every=5, v_ego=0.0):""",
                """def build_v2(core66, pose, neighbors, a_lat=0.0, steer_prev=0.0,
             step=0, replan_every=5, v_ego=0.0):""")
s1 = s1.replace("""    replan = np.array([(int(step) % max(int(replan_every), 1))
                       / float(max(int(replan_every), 1))], np.float32)
    out = np.concatenate([core[:V1_DIM - 1], ego_ex, oth_ex, replan,
                          core[V1_DIM - 1:]]).astype(np.float32)""",
                """    out = np.concatenate([core[:V1_DIM - 1], ego_ex, oth_ex,
                          core[V1_DIM - 1:]]).astype(np.float32)""")
open(p1, 'w', encoding='utf-8').write(s1.lstrip('\ufeff'))
ast.parse(open(p1, encoding='utf-8').read())
print('idc_state_v2.py OK  84 维')

# ---------- 2) agent ----------
p3 = '/gpudrive/baselines/IDC-Bridge/src/agents/idc_agent.py'
s3 = open(p3, encoding='utf-8-sig').read()
s3 = s3.replace("            self.DIM_EXTRA = 19           # ego2 + others16 + replan1（插在 veh 之前）",
                "            self.DIM_EXTRA = 18           # ego2 + others16（插在 veh 之前）")
s3 = s3.replace("""            ego_ex = s[..., 65:67]
            oth_ex = s[..., 67:83]
            replan = s[..., 83:84]
            return torch.cat([base64, ego_ex, oth_ex, replan], dim=-1)  # 83""",
                """            ego_ex = s[..., 65:67]
            oth_ex = s[..., 67:83]
            return torch.cat([base64, ego_ex, oth_ex], dim=-1)  # 82""")
s3 = s3.replace("            veil = s[..., 84:85]                                       # 1",
                "            veil = s[..., 83:84]                                       # 1")
s3 = s3.replace("""            _extra = states[:, _base:_base + self.DIM_EXTRA]   # 单步近似：原样带过去""",
                """            _extra = states[:, _base:_base + self.DIM_EXTRA]   # 单步近似：原样带过去""")
open(p3, 'w', encoding='utf-8').write(s3.lstrip('\ufeff'))
ast.parse(open(p3, encoding='utf-8').read())
print('idc_agent.py OK   TOTAL =', 6+32+8+18+1+1+18)
