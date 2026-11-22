"""把上层 Planner 的 K 条模态导出成 env 可用的外部候选（91 点世界坐标）。

与 export_head_plans.py 完全一致的参考拼法：
    ref = [日志 0..t_now 的位置] + [规划从 t_now+1 开始的 40 点] → 外推到 91 点

用法：
    python tools/export_planner_plans.py \
        --ckpt /workspace/data/ab60k_B/latest.pt \
        --files-json /workspace/data/world_files.json \
        --out /workspace/data/planner_plans.npz
"""
import argparse
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "tools"))

from metrics.openloop_pdms import Scene  # noqa: E402
from build_head_dataset import build_one  # noqa: E402
from eval_head_pdms import densify, ego_to_world  # noqa: E402
from export_head_plans import extend_to, speed_profile, N_ENV  # noqa: E402


def extend_const_speed(ref, n=N_ENV, v_min=2.0, v_max=16.0):
    """把 40 点规划按"末段朝向 + 合理速度"续到 n 点。

    不能用 extend_to（按最后一段位移外推）：规划末段常在减速，
    位移趋零 → 参考末端变成静止 → 车就爬行了。
    """
    if len(ref) >= n:
        return ref[:n]
    p = np.asarray(ref, np.float64)
    seg = np.linalg.norm(np.diff(p[-11:], axis=0), axis=1) / 0.1   # 末 1 s 速度
    v = float(np.clip(np.median(seg[seg > 0.3]) if np.any(seg > 0.3)
                      else 5.0, v_min, v_max))
    d = p[-1] - p[-2]
    h = float(np.arctan2(d[1], d[0])) if np.linalg.norm(d) > 1e-6 else 0.0
    step = np.array([np.cos(h), np.sin(h)]) * v * 0.1
    out = list(p)
    q = p[-1].copy()
    while len(out) < n:
        q = q + step
        out.append(q.copy())
    return np.asarray(out, np.float32)


def mode_ok(p40, v0):
    """参考模态的物理合理性检查（自车系 40 点，4 s）。"""
    step = np.linalg.norm(np.diff(np.vstack([[0, 0], p40]), axis=0), axis=1)
    arc = float(step.sum())
    v_mean = arc / 4.0
    if float(step.max()) > 2.5:            # 单步 >25 m/s
        return False, 'step'
    if np.abs(p40[:, 1]).max() > 10.0:      # 横向跑出 10 m
        return False, 'lat'
    if v0 >= 2.0:
        if not (0.35 * v0 <= v_mean <= 1.8 * v0 + 2.0):
            return False, 'speed'
    else:
        if arc > 6.0:
            return False, 'stop'
    return True, ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='/data/data_json/training')
    ap.add_argument('--ckpt', default='/workspace/data/ab60k_B/latest.pt')
    ap.add_argument('--num-worlds', type=int, default=12)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--k', type=int, default=6)
    ap.add_argument('--files-json', default='/workspace/data/world_files.json')
    ap.add_argument('--out', default='/workspace/data/planner_plans.npz')
    ap.add_argument('--oracle', action='store_true',
                    help='按 GT 的 FDE 排序（定责用，不是真实系统）')
    ap.add_argument('--filter', action='store_true', default=True,
                    help='过滤物理不合理的模态（默认开）')
    ap.add_argument('--no-filter', dest='filter', action='store_false')
    ap.add_argument('--gt', action='store_true',
                    help='用人类 GT 未来当候选（对照实验，验证注入接口本身）')
    ap.add_argument('--gt-full', action='store_true',
                    help='用完整 91 点人类轨迹当候选（最干净的接口对照）')
    ap.add_argument('--no-prefix', action='store_true',
                    help='参考从"当前位置"开始（第 0 点 = 现在），不拼历史前缀')
    args = ap.parse_args()

    import torch
    from train_planner import Planner
    from gpudrive.env.dataset import SceneDataLoader

    if args.files_json and os.path.exists(args.files_json):
        names = json.load(open(args.files_json))
        files = [os.path.join(args.data_dir, n) for n in names]
    else:
        loader = SceneDataLoader(root=args.data_dir,
                                 batch_size=args.num_worlds,
                                 dataset_size=args.num_worlds,
                                 sample_with_replacement=True,
                                 shuffle=True, seed=args.seed)
        idx = list(loader.indices)[:args.num_worlds]
        files = [loader.dataset[i] for i in idx]
    print('场景数:', len(files))

    ck = torch.load(args.ckpt, map_location='cpu')
    K = ck.get('K', 6)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Planner(K=K).to(dev).eval()
    model.load_state_dict(ck['model'])
    print('planner: %s  K=%d  dev=%s' % (args.ckpt, K, dev))

    zero_anchor = np.zeros((K, 8, 2), np.float32)     # 只需要标签用的占位
    out_names, paths_all, speeds_all, probs_all = [], [], [], []
    origins, origin_h = [], []
    for f in files:
        d = json.load(open(f))
        sc = Scene(d, horizon=40)
        s = build_one(d, zero_anchor)
        if s is None:
            print('  [warn] build_one failed:', os.path.basename(f))
            continue
        if args.gt_full:
            # 最干净的对照：人类轨迹（91 点，世界系）直接当候选
            if args.no_prefix:
                full = np.asarray(sc.expert_pos[sc.t_now:sc.t_now + N_ENV],
                                  np.float32)
            else:
                full = np.asarray(sc.expert_pos[:N_ENV], np.float32)
            if len(full) < N_ENV:
                full = extend_const_speed(full, N_ENV)
            sp = speed_profile(full)
            out_names.append(os.path.basename(f))
            paths_all.append(np.stack([full] * K))
            speeds_all.append(np.stack([sp] * K))
            probs_all.append(np.linspace(1.0, 0.2, K).astype(np.float32))
            origins.append(np.asarray(sc.pose_now()[:2], np.float32))
            origin_h.append(float(sc.pose_now()[2]))
            print('  %s: 用 91 点人类轨迹' % os.path.basename(f)[:22])
            continue

        xy = s['ego_world'][:2]
        h0 = float(s['ego_world'][2])
        with torch.no_grad():
            pred = model(torch.tensor(s['ego'], device=dev)[None],
                         torch.tensor(s['agent'], device=dev)[None],
                         torch.tensor(s['amask'], device=dev)[None],
                         torch.tensor(s['map'], device=dev)[None],
                         torch.tensor(s['mmask'], device=dev)[None])[0].cpu().numpy()
        gt = s['gt']
        if args.gt:
            # 对照：40 点 GT（自车系 8 点 → densify），复制 K 份
            p40_gt = densify(gt, 40)
            pred = np.stack([p40_gt] * K)
            order = list(range(K))
        fde = np.linalg.norm(pred[:, -1] - gt[-1][None], axis=-1)
        gt_arc = float(np.linalg.norm(np.diff(np.vstack(
            [[0, 0], gt]), axis=0), axis=1).sum())
        plan_arc = [float(np.linalg.norm(np.diff(np.vstack([[0, 0], densify(pred[k], 40)]),
                                                 axis=0), axis=1).sum()) for k in range(K)]
        print('  %s v0=%.1f m/s GT弧长=%.1f 预测弧长=%s FDE=%s'
              % (os.path.basename(f)[:22], float(s['ego'][0]), gt_arc,
                 np.round(plan_arc, 1), np.round(fde, 1)))
        if args.oracle:
            fde = np.linalg.norm(pred[:, -1] - gt[-1][None], axis=-1)
            order = list(np.argsort(fde))
        else:
            order = list(range(K))
        P, S, kept = [], [], []
        for k in order[:args.k]:
            p40 = densify(pred[k], 40)
            if args.filter:
                ok, why = mode_ok(p40, float(s['ego'][0]))
                if not ok:
                    continue
            dense = ego_to_world(p40, xy, h0)
            if args.no_prefix:
                ref = dense.astype(np.float32)          # 第 0 点 = 现在
            else:
                prefix = np.asarray(sc.expert_pos[:sc.t_now + 1], np.float32)
                ref = np.vstack([prefix, dense.astype(np.float32)])
            ext = extend_const_speed(ref, N_ENV)
            P.append(ext.astype(np.float32))
            S.append(speed_profile(ext))
            kept.append(int(k))
        if not P:
            print('  %s: 6 条模态全部不可执行 → 该世界回退到 IDC 内部候选'
                  % os.path.basename(f)[:22])
            continue
        while len(P) < args.k:               # 不足 K 条就复制最后一条（env 要求固定 K）
            P.append(P[-1]); S.append(S[-1])
        print('  %s 保留模态 %s（原 %d 条）'
              % (os.path.basename(f)[:22], kept, args.k))
        out_names.append(os.path.basename(f))
        paths_all.append(np.stack(P))
        speeds_all.append(np.stack(S))
        probs_all.append(np.linspace(1.0, 0.2, args.k).astype(np.float32))
        origins.append(np.asarray(sc.pose_now()[:2], np.float32))
        origin_h.append(float(sc.pose_now()[2]))
        print('  %s: 6 条模态终点弧长 %s'
              % (os.path.basename(f)[:24],
                 np.round([np.linalg.norm(np.diff(p, axis=0), axis=1).sum()
                           for p in P], 1)))

    np.savez_compressed(args.out, names=np.array(out_names),
                        paths=np.stack(paths_all),
                        probs=np.stack(probs_all),
                        speeds=np.stack(speeds_all),
                        origin_pos=np.stack(origins),
                        origin_heading=np.asarray(origin_h, np.float32))
    print('saved', args.out, 'worlds', len(out_names))


if __name__ == '__main__':
    main()
