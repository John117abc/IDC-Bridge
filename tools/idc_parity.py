"""修正后的双侧一致性测试。"""
import math
import sys

import numpy as np

sys.path.insert(0, '/gpudrive/baselines/IDC-Bridge/tools')
import frames as F  # noqa: E402

DP_IDX = F.DIM_EGO + F.DIM_OTHERS + F.DIM_VALIDITY      # 46


def mk_scene(seed=0):
    r = np.random.default_rng(seed)
    pose = (float(r.uniform(-5, 5)), float(r.uniform(-5, 5)), float(r.uniform(-3, 3)))
    ego_v = float(r.uniform(2, 15))
    nbrs = []
    for _ in range(8):
        d = float(r.uniform(5, 30))
        ang = float(r.uniform(-math.pi, math.pi))
        nbrs.append(dict(x=pose[0] + d * math.cos(ang),
                         y=pose[1] + d * math.sin(ang),
                         h=float(r.uniform(-math.pi, math.pi)),
                         v=float(r.uniform(0, 15))))
    nbrs.sort(key=lambda q: (q['x'] - pose[0]) ** 2 + (q['y'] - pose[1]) ** 2)
    ref = (pose[0] + 10 * math.cos(pose[2]), pose[1] + 10 * math.sin(pose[2]))
    return pose, ego_v, nbrs, ref


def test_same_physical_scene():
    print('=== 1) 同一物理场景：CARLA 表示 → canonical vs Waymo 表示 ===')
    worst = 0.0
    for seed in range(5):
        pose, v, nbrs, ref = mk_scene(seed)
        ref_h = pose[2]
        st_w = F.build_state84(pose, v, nbrs, ref, ref_h, 8.0, ref_idx=7)

        # 把场景写成 CARLA 原始量，再按约定转回 canonical
        cx, cy, cyaw = F.canonical_to_carla(*pose)
        pose_c = F.carla_to_canonical(cx, cy, cyaw)
        nbrs_c = []
        for nb in nbrs:
            nx, ny, nyaw = F.canonical_to_carla(nb['x'], nb['y'], nb['h'])
            bx, by, bh = F.carla_to_canonical(nx, ny, nyaw)
            nbrs_c.append(dict(x=bx, y=by, h=bh, v=nb['v']))
        rx, ry, ryaw = F.canonical_to_carla(ref[0], ref[1], ref_h)
        ref_c = F.carla_to_canonical(rx, ry, ryaw)
        ref_h_c = ref_c[2]
        st_c = F.build_state84(pose_c, v, nbrs_c, ref_c[:2], ref_h_c, 8.0, ref_idx=7)

        d = float(np.abs(st_w - st_c).max())
        e = float(np.abs(F.encode82(st_w) - F.encode82(st_c)).max())
        worst = max(worst, max(d, e))
        print(f'  seed={seed}: |Δ state84| = {d:.2e}   |Δ encode82| = {e:.2e}')
    ok = worst < 1e-4
    print(f'  最大逐维差 = {worst:.2e}   {"通过" if ok else "不通过"}')
    return ok


def test_mirror_detection():
    print('\n=== 2) 镜像检测（确认测试有区分能力）===')
    pose, v, nbrs, ref = mk_scene(1)
    st_ok = F.build_state84(pose, v, nbrs, ref, pose[2], 8.0)
    bad = [dict(x=nb['x'], y=-nb['y'], h=-nb['h'], v=nb['v']) for nb in nbrs]
    st_bad = F.build_state84((pose[0], pose[1], pose[2]), v, bad,
                             (ref[0], -ref[1]), -pose[2], 8.0)
    d = float(np.abs(F.encode82(st_ok) - F.encode82(st_bad)).max())
    print(f'  镜像后 encode82 最大差 = {d:.2f}（应 > 0.1）')
    return d > 0.1


def test_sign_semantics():
    print('\n=== 3) 语义：参考在左侧 dp 应为正 ===')
    st_l = F.build_state84((0., 0., 0.), 5.0, [], (10., 2.), 0.0, 5.0)
    st_r = F.build_state84((0., 0., 0.), 5.0, [], (10., -2.), 0.0, 5.0)
    dp_l, dp_r = float(st_l[DP_IDX]), float(st_r[DP_IDX])
    print(f'  参考在左 dp={dp_l:+.3f}   参考在右 dp={dp_r:+.3f}')
    ok = dp_l > 0 > dp_r
    print('  通过' if ok else '  不通过')
    return ok


if __name__ == '__main__':
    r = [test_same_physical_scene(), test_mirror_detection(), test_sign_semantics()]
    print('\n结论:', 'ALL PASS' if all(r) else 'FAIL')
