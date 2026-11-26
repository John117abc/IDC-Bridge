"""修纵向：off_acc 分支的 acc 从 0 积分会让车永远起步不了

加开关 VEC_IDC_LONG_REF=1：v_tgt = clip(v_ref + 0.15·acc)，不再积分
"""
import io

PATHS = [
    '/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/tools/plant_agent_idc.py',
    '/home/jiangchengxuan/PycharmProjects/research-safe-drive/plant2/carla_garage/plant_agent_idc.py',
]

OLD = """                if getattr(self.pol, 'action_space', 'delta_acc') == 'off_acc':
                    # 2D 动作空间：加速度意图积分成速度指令（v_cmd += a·dt）
                    _dt = 0.05 * self.every
                    _vmin = float(os.environ.get('VEC_IDC_VMIN', '0.0'))
                    _vc = float(getattr(self, '_v_cmd', None) or _vmin)
                    _vc = float(np.clip(_vc + acc * _dt, _vmin, 20.0))
                    self._v_cmd = _vc
                    _v_raw = _vc"""

NEW = """                if getattr(self.pol, 'action_space', 'delta_acc') == 'off_acc':
                    # 2D 动作空间
                    _vmin = float(os.environ.get('VEC_IDC_VMIN', '0.0'))
                    if os.environ.get('VEC_IDC_LONG_REF', '1') == '1':
                        # 参考速度 + acc 小幅修正（不积分，避免 acc≈0 时永远起步不了）
                        _v_ref0 = float(_v_raw)
                        _v_raw = float(np.clip(_v_ref0 + _vg * float(acc),
                                               _vmin, 20.0))
                    else:
                        _dt = 0.05 * self.every
                        _vc = float(getattr(self, '_v_cmd', None) or _vmin)
                        _vc = float(np.clip(_vc + acc * _dt, _vmin, 20.0))
                        self._v_cmd = _vc
                        _v_raw = _vc"""

for p in PATHS:
    s = io.open(p, encoding='utf-8', errors='replace').read()
    if 'VEC_IDC_LONG_REF' in s:
        print('已有:', p)
        continue
    if OLD not in s:
        print('锚点未匹配:', p)
        continue
    s = s.replace(OLD, NEW, 1)
    io.open(p, 'w', encoding='utf-8', newline='').write(s)
    print('已修纵向:', p)
