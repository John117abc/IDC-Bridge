"""冒烟测试：actor 结构、维度、权重往返、agent 可导入。"""
import sys

import torch
import yaml

sys.path.insert(0, '/gpudrive/baselines/IDC-Bridge/src')
from models.continuous_actor_critic import (PaperMLPActor, TransformerActor,
                                            ContinuousCritic)

cfg = yaml.safe_load(open('/gpudrive/baselines/IDC-Bridge/src/configs/base.yaml'))
a = cfg.get('agent', {})
print('config: actor_type =', a.get('actor_type'), '| hidden_dim =',
      a.get('hidden_dim'), '| ref_window =', a.get('ref_window'))

for D in (62, 51, 44):
    m = PaperMLPActor(D, 256)
    y = m(torch.randn(8, D))
    n = sum(p.numel() for p in m.parameters())
    m2 = PaperMLPActor(D, 256)
    m2.load_state_dict(m.state_dict())
    print(f'MLP  D={D:3d} out={tuple(y.shape)} params={n:6d} roundtrip=ok')

t = TransformerActor(state_dim=62, window_size=16)
y = t(torch.randn(8, 16, 62))
print('Transformer out=', tuple(y.shape), 'params=',
      sum(p.numel() for p in t.parameters()))
c = ContinuousCritic(62, 512)
print('Critic out=', tuple(c(torch.randn(8, 62)).shape))

try:
    import agents.idc_agent as ag
    print('idc_agent import OK; has actor_type switch:',
          'PaperMLPActor' in open(ag.__file__, encoding='utf-8').read())
except Exception as e:
    print('idc_agent import FAILED:', type(e).__name__, e)

# ---- 配置解析检查（真实 build_config 路径）----
try:
    from utils.config import build_config
    from agents.idc_agent import _cfgget
    c = build_config('/gpudrive/baselines/IDC-Bridge/src/configs/base.yaml')
    print('build_config ->', type(c).__name__, '| keys:', list(vars(c))[:8])
    print('resolved:', {k: _cfgget(c, k, None) for k in
                        ('actor_type', 'actor_hidden', 'critic_hidden')})
except Exception as e:
    print('config check FAILED:', type(e).__name__, e)
