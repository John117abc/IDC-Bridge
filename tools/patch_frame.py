"""修 planner_features/planner_runtime 的坐标系不一致：
   json 是世界坐标，GPUDrive 的 sim 坐标 = 世界坐标 − world_means。
"""
import ast
import io

R = '/gpudrive/baselines/IDC-Bridge/src/env'

# ---------- 1) ScenePack 增加 shift ----------
pf = f'{R}/planner_features.py'
src = open(pf).read()
if 'def shift(self' not in src:
    anchor = "        self.seg_lane = np.asarray(lane, np.float64) if lane else np.zeros((0,))\n"
    assert anchor in src, '找不到 seg_lane 锚点'
    add = anchor + '''
    def shift(self, dx, dy):
        """整体平移（世界坐标 → GPUDrive sim 坐标，差一个逐场景均值）。"""
        d = np.array([dx, dy], np.float64)
        if len(self.seg_a):
            self.seg_a = self.seg_a + d
            self.seg_b = self.seg_b + d
        for a in self.agents:
            a['pos'] = a['pos'] + d
        if getattr(self, 'ego_pos', None) is not None and len(self.ego_pos):
            self.ego_pos = self.ego_pos + d
'''
    src = src.replace(anchor, add, 1)
    open(pf, 'w').write(src)
ast.parse(open(pf).read())
print('PATCH1 OK  ScenePack.shift =', 'def shift(self' in open(pf).read())

# ---------- 2) _ensure_pack 里平移 ----------
pr = f'{R}/planner_runtime.py'
src = open(pr).read()
old = """            pk = load_pack(str(env.data_loader.dataset[idx]))"""
new = """            pk = load_pack(str(env.data_loader.dataset[idx]))
            # json 是世界坐标；sim 坐标 = 世界 − world_means（逐场景）
            try:
                _m = env.sim.world_means_tensor().to_torch().cpu().numpy()[w, :2]
                pk.shift(-float(_m[0]), -float(_m[1]))
            except Exception as _me:
                print(f'[RT] world{w} shift 失败: {_me}', flush=True)"""
if 'world_means_tensor' not in src:
    assert old in src, '找不到 load_pack 锚点'
    src = src.replace(old, new, 1)
    open(pr, 'w').write(src)
ast.parse(open(pr).read())
print('PATCH2 OK  shift 调用 =', 'world_means_tensor' in open(pr).read())
