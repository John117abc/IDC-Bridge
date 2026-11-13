"""统计 IDC-Bridge 网络输入（65 维 state）每一维的真实取值范围。

用法（容器内）：
    python tools/stat_state.py --data-dir /data/data_json/training
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from gpudrive.env.env_torch import GPUDriveTorchEnv          # noqa: E402
from gpudrive.env.dataset import SceneDataLoader             # noqa: E402
from env.env_utils import get_env_config, load_scenes, get_ego_indices  # noqa: E402
from env.idc_state_builder import GPUDriveObservationBuilder  # noqa: E402

DIM_EGO, DIM_OTH, DIM_VAL, DIM_REF, DIM_T = 6, 32, 8, 18, 1


def dim_names():
    n = ['ego_x', 'ego_y', 'v_lon', 'v_lat', 'phi', 'omega']
    for i in range(8):
        n += [f'oth{i}_x', f'oth{i}_y', f'oth{i}_phi', f'oth{i}_v']
    n += [f'valid{i}' for i in range(8)]
    n += ['dp', 'dphi', 'dv']
    for tag in ('l1', 'l2', 'l3'):
        n += [f'{tag}_lat', f'{tag}_dphi', f'{tag}_road', f'{tag}_spd']
    n += ['curv_ahead', 'spd_ahead', 'road_ahead']
    n += ['temporal']
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--num-worlds', type=int, default=8)
    ap.add_argument('--dataset-size', type=int, default=32)
    ap.add_argument('--steps', type=int, default=25)
    args = ap.parse_args()

    loader = SceneDataLoader(root=args.data_dir, batch_size=args.num_worlds,
                             dataset_size=args.dataset_size,
                             sample_with_replacement=False, shuffle=True, seed=5)
    env = GPUDriveTorchEnv(config=get_env_config(), data_loader=loader,
                           max_cont_agents=1, device='cuda',
                           action_type='continuous')
    env.reset()
    builder = GPUDriveObservationBuilder(env, load_scenes(env))
    ego = get_ego_indices(env, args.num_worlds)
    builder.generate_candidate_paths(ego, num_paths=1, use_expert_path=True)

    rows = []
    for _ in range(args.steps):
        ss = builder.get_idc_observations_batch(ego, path_indices=[0] * args.num_worlds)
        rows.append(np.stack(ss))
        for w in range(args.num_worlds):
            builder.increment_step(w)
        a = torch.zeros(args.num_worlds, 1, 3, device='cuda')   # [acc, steer, head]
        a[..., 0] = 0.6          # 给一点油门，让车动起来
        a[..., 1] = 0.05
        env.step_dynamics(a)

    X = np.concatenate(rows, 0).astype(np.float64)
    names = dim_names()
    assert X.shape[1] == len(names), (X.shape, len(names))

    print(f'样本数 = {X.shape[0]}  (worlds={args.num_worlds} x steps={args.steps})')
    print(f"{'dim':14s} {'name':12s} {'min':>10s} {'max':>10s} {'std':>10s} {'|max|/std':>10s}")
    print('-' * 72)
    for i, nm in enumerate(names):
        col = X[:, i]
        sd = col.std()
        ratio = abs(col).max() / sd if sd > 1e-9 else float('inf')
        print(f'{i:3d}  {nm:20s} {col.min():10.3f} {col.max():10.3f} '
              f'{sd:10.3f} {ratio:10.1f}')

    print('\n--- 按块汇总 ---')
    blocks = [('ego', 0, 6), ('others', 6, 38), ('validity', 38, 46),
              ('ref_error', 46, 64), ('temporal', 64, 65)]
    for tag, XX in (('全部', X),):
        print(f'[{tag}] 样本 {XX.shape[0]}')
        for nm, a0, a1 in blocks:
            seg = XX[:, a0:a1]
            print(f'  {nm:10s} 绝对值最大 = {abs(seg).max():10.2f}   '
                  f'标准差范围 = [{seg.std(0).min():.4f}, {seg.std(0).max():.3f}]')

    # 去掉训��里会被 WorldManager 判为 bad 的 world（|ego| > filter_threshold=200），
    # 看"干净"输入的真实量纲
    keep = (np.abs(X[:, 0]) < 200) & (np.abs(X[:, 1]) < 200)
    Y = X[keep]
    print(f'\n--- 过滤 |ego_x|,|ego_y| < 200 之后（保留 {Y.shape[0]}/{X.shape[0]} 行）---')
    for nm, a0, a1 in blocks:
        seg = Y[:, a0:a1]
        print(f'  {nm:10s} 绝对值最大 = {abs(seg).max():10.2f}   '
              f'标准差范围 = [{seg.std(0).min():.4f}, {seg.std(0).max():.3f}]')

    sds = np.array([Y[:, i].std() for i in range(Y.shape[1])])
    sds_pos = sds[sds > 1e-6]
    print(f'\n[过滤后] 各维标准差跨度: {sds_pos.min():.4f} ~ {sds_pos.max():.1f} '
          f'(相差 {sds_pos.max() / sds_pos.min():.0f} 倍)')

    print('\n[过滤后] 每维标准差（绝对值从大到小）:')
    order = np.argsort(-sds)
    for i in order[:12]:
        print(f'  {i:3d}  {names[i]:14s} std={sds[i]:10.3f}  '
              f'range=[{Y[:, i].min():8.3f}, {Y[:, i].max():8.3f}]')
    print('  ...')
    for i in order[-8:]:
        print(f'  {i:3d}  {names[i]:14s} std={sds[i]:10.5f}  '
              f'range=[{Y[:, i].min():8.4f}, {Y[:, i].max():8.4f}]')

    # 线性层输入尺度跨度
    all_sds = np.array([X[:, i].std() for i in range(X.shape[1])])
    all_pos = all_sds[all_sds > 1e-6]
    print(f'\n[全部] 各维标准差跨度: {all_pos.min():.4f} ~ {all_pos.max():.1f} '
          f'(相差 {all_pos.max() / all_pos.min():.0f} 倍)')


if __name__ == '__main__':
    main()
