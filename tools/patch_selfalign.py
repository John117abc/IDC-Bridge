"""改用"自车观测自对齐"平移 pack（不依赖 world_means 的定义）。"""
import ast

pr = '/gpudrive/baselines/IDC-Bridge/src/env/planner_runtime.py'
src = open(pr).read()

# 1) _ensure_pack 增加 step 参数 + 自对齐
old_sig = "    def _ensure_pack(self, env, w):"
new_sig = "    def _ensure_pack(self, env, w, step=0):"
if old_sig in src:
    src = src.replace(old_sig, new_sig, 1)

old_shift = """            # json 是世界坐标；sim 坐标 = 世界 − world_means（逐场景）
            try:
                _m = env.sim.world_means_tensor().to_torch().cpu().numpy()[w, :2]
                pk.shift(-float(_m[0]), -float(_m[1]))
            except Exception as _me:
                print(f'[RT] world{w} shift 失败: {_me}', flush=True)"""
new_shift = """            # 坐标对齐：用自车观测与 pack 自身 ego_pos 之差（与坐标系定义无关）
            try:
                _obs = self.buf[w][-1]
                _t = int(min(10 + int(step), len(pk.ego_pos) - 1))
                _ref = pk.ego_pos[_t]
                pk.shift(float(_obs[0]) - float(_ref[0]),
                         float(_obs[1]) - float(_ref[1]))
                if getattr(self, 'debug', False):
                    print('[RT][align] w=%d obs=(%.1f,%.1f) pack_ego=(%.1f,%.1f) '
                          'seg=%d' % (w, float(_obs[0]), float(_obs[1]),
                                      float(_ref[0]), float(_ref[1]),
                                      len(pk.seg_a)), flush=True)
            except Exception as _me:
                print(f'[RT] world{w} 对齐失败: {_me}', flush=True)"""
if 'world_means_tensor' in src:
    src = src.replace(old_shift, new_shift, 1)

# 2) 调用处传 step
src = src.replace("pk = self._ensure_pack(env, w)",
                  "pk = self._ensure_pack(env, w, step)", 1)
open(pr, 'w').write(src)
ast.parse(open(pr).read())
print('PATCH OK  align =', 'RT][align' in open(pr).read(),
      ' call_step =', '_ensure_pack(env, w, step)' in open(pr).read())
