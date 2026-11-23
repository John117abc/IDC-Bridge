"""修 actor 构造参数（签名：state_dim, d_model, nhead, num_layers, window_size, dropout）。"""
import ast

f = '/gpudrive/baselines/IDC-Bridge/tools/idc_carla.py'
s = open(f, encoding='utf-8-sig').read()

old = """        self.actor = TransformerActor(
            state_dim=82, hidden_dim=hidden, d_model=d_model,
            nhead=nhead, num_layers=layers, dropout=drop).to(self.dev).eval()"""
new = """        win = int(getattr(cfg, 'window_size', WINDOW) or WINDOW)
        self.actor = TransformerActor(
            state_dim=82, d_model=d_model, nhead=nhead,
            num_layers=layers, window_size=win, dropout=drop).to(self.dev).eval()
        print(f'[idc] actor: d_model={d_model} nhead={nhead} layers={layers} '
              f'window={win}', flush=True)"""
assert old in s
s = s.replace(old, new, 1)
open(f, 'w', encoding='utf-8').write(s.lstrip('\ufeff'))
ast.parse(open(f, encoding='utf-8').read())
print('PATCH OK')
