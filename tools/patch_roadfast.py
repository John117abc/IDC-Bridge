"""加速：road_dist 改用 KD-tree（原来暴力遍历全城车道段，导致 TickRuntime）。"""
import ast

f = '/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/tools/plant_agent_idc.py'
s = open(f, encoding='utf-8-sig').read()

# 1) 构造参数加 tree
old_init = """    def __init__(self, ckpt, world, vehicle, map_npz, infer_every=2):"""
new_init = """    def __init__(self, ckpt, world, vehicle, map_npz, tree=None, infer_every=2):"""
s = s.replace(old_init, new_init, 1)
s = s.replace("        self.seg_a, self.seg_b = _load_segments(map_npz)",
              "        self.seg_a, self.seg_b = _load_segments(map_npz)\n        self.tree = tree", 1)

# 2) _path_canon：跳过硬算，改用 tree
old_path = """        p = S.build_path(np.asarray(plan_ego, np.float64), pose_carla,
                         self.seg_a, self.seg_b)"""
new_path = """        # 传 None 跳过 S.build_path 内部的暴力 road_dist，下面用 KD-tree 算
        p = S.build_path(np.asarray(plan_ego, np.float64), pose_carla, None, None)"""
assert old_path in s
s = s.replace(old_path, new_path, 1)

old_ret = """        return dict(pos=pos_c.astype(np.float32), heading=hd_c,
                    speed=np.asarray(p['speed'], np.float32),
                    road_dist=np.asarray(p['road_dist'], np.float32),"""
new_ret = """        road = np.asarray(p['road_dist'], np.float32)
        if self.tree is not None:
            try:
                _d, _ = self.tree.query(np.asarray(p['pos'], np.float64), k=1)
                road = np.minimum(np.asarray(_d, np.float32), 200.0)
            except Exception as _e:
                print(f'[IDC] road_dist 查询失败: {_e}', flush=True)
        return dict(pos=pos_c.astype(np.float32), heading=hd_c,
                    speed=np.asarray(p['speed'], np.float32),
                    road_dist=road,"""
assert old_ret in s
s = s.replace(old_ret, new_ret, 1)

open(f, 'w', encoding='utf-8').write(s.lstrip('\ufeff'))
ast.parse(open(f, encoding='utf-8').read())
print('PATCH OK')
