"""修纵向抖动/急刹：

1) carla_pid_exec.LongitudinalPID
   · 取消"超过目标 3.2% 就满刹"，改成比例刹车
   · KP/KD/KI/ratio 支持 env 覆盖，默认降 KD、KI
2) plant_agent_idc：v_tgt 加变化率限制（每 0.2 s 最多 ±1.5 m/s）
"""
import io

PE = ['/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/tools/carla_pid_exec.py',
      '/home/jiangchengxuan/PycharmProjects/research-safe-drive/plant2/carla_garage/carla_pid_exec.py']
IDC = ['/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/tools/plant_agent_idc.py',
       '/home/jiangchengxuan/PycharmProjects/research-safe-drive/plant2/carla_garage/plant_agent_idc.py']

OLD = """        tgt = max(LON_MIN_TARGET, float(target_speed))
        v_now, v_tgt = 3.6 * float(speed), 3.6 * tgt      # km/h
        if v_tgt > 1e-3 and v_now / v_tgt > LON_BRAKE_RATIO:
            return 0.0, 1.0
        err = v_tgt - v_now
        d_err = 0.0 if self.prev_err is None else (err - self.prev_err) / max(dt, 1e-3)
        self.prev_err = err
        u = LON_KP * err + LON_KD * d_err + LON_KI * np.clip(err * dt, -5, 5)
        # 油门标定：err=10 km/h 时约 0.5 油门（原来除以 100 太弱，车只能爬行）
        gain = float(os.environ.get('VEC_LON_SCALE', '20.0'))
        throttle = float(np.clip(u / gain, 0.0, 1.0))
        brake = float(np.clip(-u / gain, 0.0, 1.0)) if u < -2.0 else 0.0
        return throttle, brake"""

NEW = """        tgt = max(LON_MIN_TARGET, float(target_speed))
        v_now, v_tgt = 3.6 * float(speed), 3.6 * tgt      # km/h
        kp = float(os.environ.get('VEC_LON_KP', '0.8'))
        kd = float(os.environ.get('VEC_LON_KD', '0.25'))
        ki = float(os.environ.get('VEC_LON_KI', '0.05'))
        ratio = float(os.environ.get('VEC_LON_RATIO', '1.10'))
        err = v_tgt - v_now
        d_err = 0.0 if self.prev_err is None else (err - self.prev_err) / max(dt, 1e-3)
        self.prev_err = err
        u = kp * err + kd * d_err + ki * np.clip(err * dt, -5, 5)
        gain = float(os.environ.get('VEC_LON_SCALE', '20.0'))
        throttle = float(np.clip(u / gain, 0.0, 1.0))
        # 比例刹车：越过目标 ratio 倍后平滑加压，不再满刹
        brake = 0.0
        if v_tgt > 1e-3:
            over = v_now / v_tgt - ratio
            if over > 0:
                brake = float(np.clip(over * 3.0, 0.0, 1.0))
        if u < -2.0:
            brake = max(brake, float(np.clip(-u / gain, 0.0, 1.0)))
        return throttle, brake"""

for p in PE:
    s = io.open(p, encoding='utf-8', errors='replace').read()
    if 'VEC_LON_RATIO' in s:
        print('PE 已修:', p)
        continue
    if OLD not in s:
        print('PE 锚点未匹配:', p)
        continue
    s = s.replace(OLD, NEW, 1)
    io.open(p, 'w', encoding='utf-8', newline='').write(s)
    print('PE 已修:', p)

# ---- v_tgt 变化率限制 ----
OLD2 = """                _prev = getattr(self, '_v_tgt_prev', None)
                v_tgt = _v_raw if _prev is None else 0.8 * _prev + 0.2 * _v_raw"""
NEW2 = """                _prev = getattr(self, '_v_tgt_prev', None)
                v_tgt = _v_raw if _prev is None else 0.8 * _prev + 0.2 * _v_raw
                # 变化率限制：0.2 s 内最多 ±1.5 m/s，避免目标速度跳变引发急刹
                if _prev is not None:
                    _dmax = float(os.environ.get('VEC_VTGT_DMAX', '1.5'))
                    v_tgt = float(np.clip(v_tgt, _prev - _dmax, _prev + _dmax))"""
for p in IDC:
    s = io.open(p, encoding='utf-8', errors='replace').read()
    if 'VEC_VTGT_DMAX' in s:
        print('IDC 已修:', p)
        continue
    if OLD2 not in s:
        print('IDC 锚点未匹配:', p)
        continue
    s = s.replace(OLD2, NEW2, 1)
    io.open(p, 'w', encoding='utf-8', newline='').write(s)
    print('IDC 已修:', p)
