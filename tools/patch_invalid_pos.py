"""修 ScenePack：Waymo 用 -10000 作无效帧占位，需按 valid 掩码补齐。"""
import ast

pf = '/gpudrive/baselines/IDC-Bridge/src/env/planner_features.py'
src = open(pf).read()

anchor = """            self.agents.append(dict(
                pos=np.asarray([[p['x'], p['y']] for p in q], np.float64),
                head=np.asarray(o.get('heading') or [0.0] * len(q), np.float64),
                valid=np.asarray(o.get('valid') or [1] * len(q)),"""
new = """            _pos = np.asarray([[p['x'], p['y']] for p in q], np.float64)
            _val = np.asarray(o.get('valid') or [1] * len(q), np.float64)
            # Waymo 用 -10000/无效帧占位：按 valid 掩码把无效帧替换为最近有效位置
            _bad = (_val <= 0.5) | (np.abs(_pos) > 1e4).any(axis=1)
            if _bad.any():
                _ok = np.where(~_bad)[0]
                if len(_ok):
                    _pos[_bad] = _pos[_ok[0]]
                else:
                    _val[:] = 0.0
            self.agents.append(dict(
                pos=_pos,
                head=np.asarray(o.get('heading') or [0.0] * len(q), np.float64),
                valid=_val,"""
if '_bad = (_val <= 0.5)' not in src:
    assert anchor in src, '找不到 agents.append 锚点'
    src = src.replace(anchor, new, 1)
    open(pf, 'w').write(src)
ast.parse(open(pf).read())
print('PATCH OK  masking =', '_bad = (_val <= 0.5)' in open(pf).read())
