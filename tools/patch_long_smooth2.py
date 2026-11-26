"""第二轮纵向平滑：低通 0.9/0.1，变化率 ≤0.8 m/s"""
import io

IDC = ['/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/tools/plant_agent_idc.py',
       '/home/jiangchengxuan/PycharmProjects/research-safe-drive/plant2/carla_garage/plant_agent_idc.py']

OLD = """                _prev = getattr(self, '_v_tgt_prev', None)
                v_tgt = _v_raw if _prev is None else 0.8 * _prev + 0.2 * _v_raw
                # 变化率限制：0.2 s 内最多 ±1.5 m/s，避免目标速度跳变引发急刹
                if _prev is not None:
                    _dmax = float(os.environ.get('VEC_VTGT_DMAX', '1.5'))
                    v_tgt = float(np.clip(v_tgt, _prev - _dmax, _prev + _dmax))"""

NEW = """                _prev = getattr(self, '_v_tgt_prev', None)
                _a = float(os.environ.get('VEC_VTGT_A', '0.9'))
                v_tgt = _v_raw if _prev is None else _a * _prev + (1 - _a) * _v_raw
                if _prev is not None:
                    _dmax = float(os.environ.get('VEC_VTGT_DMAX', '0.8'))
                    v_tgt = float(np.clip(v_tgt, _prev - _dmax, _prev + _dmax))"""

for p in IDC:
    s = io.open(p, encoding='utf-8', errors='replace').read()
    if '_a = float(os.environ.get' in s:
        print('已改:', p)
        continue
    if OLD not in s:
        print('锚点未匹配:', p)
        continue
    s = s.replace(OLD, NEW, 1)
    io.open(p, 'w', encoding='utf-8', newline='').write(s)
    print('已改:', p)
