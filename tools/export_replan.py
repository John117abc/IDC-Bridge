"""导出"滚动重规划"候选（含 frames 维），配合 idc-eval --replan-candidates 使用。

每个重规划帧 f：
    绝对帧 t = 10 + f（10 = Waymo 的决策帧）
    在 t 处取规划器（或 GT）的 40 点未来 → 世界坐标
    参考 = [t 处自车位姿] + 40 点 → 外推到 91 点   （第 0 点 = 当前，41 点有效）
env 侧会用 target_pos 把第 0 点重新锚定到仿真里自车的真实位置。

用法：
    # 单帧（验证接口）
    python tools/export_replan.py --gt --frames 0 --out /workspace/data/replan_gt0.npz
    # 每 0.5 s 重规划，整段
    python tools/export_replan.py --ckpt .../latest.pt --frames 0,5,10,...,85
"""
import argparse
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "tools"))

from build_head_dataset import build_one, HIST  # noqa: E402
from eval_head_pdms import densify, ego_to_world  # noqa: E402

N_ENV = 91
T_NOW = 10


def extend_const_speed(ref, n=N_ENV, v_min=1.0, v_max=16.0):
    if len(ref) >= n:
        return np.asarray(ref[:n], np.float32)
    p = np.asarray(ref, np.float64)
    seg = np.linalg.norm(np.diff(p[-11:], axis=0), axis=1) / 0.1
    v = float(np.clip(np.median(seg[seg > 0.3]) if np.any(seg > 0.3) else 5.0,
                      v_min, v_max))
    d = p[-1] - p[-2]
    h = float(np.arctan2(d[1], d[0])) if np.linalg.norm(d) > 1e-6 else 0.0
    step = np.array([np.cos(h), np.sin(h)]) * v * 0.1
    out = list(p)
    q = p[-1].copy()
    while len(out) < n:
        q = q + step
        out.append(q.copy())
    return np.asarray(out, np.float32)


def speed_profile(path, dt=0.1):
    if len(path) < 2:
        return np.full(len(path), 0.5, np.float32)
    v = np.linalg.norm(np.diff(path, axis=0), axis=1) / dt
    v = np.r_[v, v[-1]]
    return np.maximum(v, 0.5).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='/data/data_json/training')
    ap.add_argument('--files-json', default='/workspace/data/world_files.json')
    ap.add_argument('--ckpt', default='/workspace/data/ab60k_B/latest.pt')
    ap.add_argument('--gt', action='store_true', help='用 GT 未来（接口对照）')
    ap.add_argument('--frames', default='0',
                    help='重规划帧（相对决策帧），逗号分隔，如 0,10,20')
    ap.add_argument('--k', type=int, default=1)
    ap.add_argument('--speed-mode', default='const',
                    choices=['const', 'profile'],
                    help='const=match env 内部参考的恒定速度（默认）；profile=按位移算')
    ap.add_argument('--speed-val', type=float, default=20.0,
                    help='--speed-mode const 时用的速度（env 内部参考是 20）')
    ap.add_argument('--out', default='/workspace/data/replan.npz')
    args = ap.parse_args()

    files = [os.path.join(args.data_dir, n)
             for n in json.load(open(args.files_json))]
    frames = [int(x) for x in args.frames.split(',') if x.strip() != '']

    import torch
    from train_planner import Planner
    K = args.k
    model = None
    if not args.gt:
        ck = torch.load(args.ckpt, map_location='cpu')
        K = args.k if args.k > 1 else ck.get('K', 6)
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = Planner(K=K).to(dev).eval()
        model.load_state_dict(ck['model'])
        print('planner %s K=%d dev=%s' % (args.ckpt, K, dev))
    K = args.k if args.k >= 1 else K

    names, paths, speeds, origins, origin_h = [], [], [], [], []
    for f in files:
        d = json.load(open(f))
        pos = np.asarray([[q['x'], q['y']] for q in
                          ((d['objects'][int(d['metadata']['sdc_track_index'])])
                           .get('position') or [])], np.float64)
        head = np.asarray(d['objects'][int(d['metadata']['sdc_track_index'])]
                          .get('heading') or [0.0] * len(pos), np.float64)
        W = np.zeros((len(frames), K, N_ENV, 2), np.float32)
        S = np.zeros((len(frames), K, N_ENV), np.float32)
        for fi, fr in enumerate(frames):
            t = T_NOW + fr
            if t + 1 >= len(pos):
                t = len(pos) - 2
            p0, h0 = pos[t], float(head[t])
            if args.gt:
                gt = np.asarray([[q['x'], q['y']] for q in
                                 d['objects'][int(d['metadata']['sdc_track_index'])]['position']],
                                np.float64)
                idx = [min(t + 5 * (i + 1), len(gt) - 1) for i in range(8)]
                fut = gt[idx]
                # 自车系 8 点 → 40 点
                rel = fut - p0
                c, s = np.cos(-h0), np.sin(-h0)
                rel = np.stack([rel[:, 0] * c - rel[:, 1] * s,
                                rel[:, 0] * s + rel[:, 1] * c], 1)
                p40 = densify(rel, 40)
                preds = np.stack([p40] * K)
            else:
                smp = build_one(d, np.zeros((K, 8, 2), np.float32), t_now=t)
                if smp is None:
                    print('  [warn] build_one 失败 t=%d %s' % (t, os.path.basename(f)))
                    break
                with torch.no_grad():
                    pr = model(torch.tensor(smp['ego'], device=dev)[None],
                               torch.tensor(smp['agent'], device=dev)[None],
                               torch.tensor(smp['amask'], device=dev)[None],
                               torch.tensor(smp['map'], device=dev)[None],
                               torch.tensor(smp['mmask'], device=dev)[None])[0]
                preds = pr.cpu().numpy()
            for k in range(K):
                # preds[k] 已经是 40 点（0.1 s），不能再 densify（那是 8 点→40 点的插值）
                dw = ego_to_world(preds[k], p0, h0)
                ref = np.vstack([p0[None], dw])
                ext = extend_const_speed(ref, N_ENV)
                W[fi, k] = ext
                S[fi, k] = (np.full(N_ENV, args.speed_val, np.float32)
                            if args.speed_mode == 'const' else speed_profile(ext))
        names.append(os.path.basename(f))
        paths.append(W)
        speeds.append(S)
        origins.append(np.asarray(pos[0], np.float32))
        d0 = pos[1] - pos[0]
        origin_h.append(float(np.arctan2(d0[1], d0[0])) if np.linalg.norm(d0) > 1e-6 else 0.0)
        print('  %s: %d 帧 × %d 模态' % (os.path.basename(f)[:22], len(frames), K))

    np.savez_compressed(args.out, names=np.array(names),
                        paths=np.stack(paths), speeds=np.stack(speeds),
                        frames=np.array(frames),
                        origin_pos=np.stack(origins),
                        origin_heading=np.asarray(origin_h, np.float32))
    print('saved', args.out, 'worlds', len(names), 'frames', frames)


if __name__ == '__main__':
    main()
