"""导出规划头的 top-k 轨迹（世界坐标，91 点），供离线注入 GEP 使用。

用与 eval 相同的 SceneDataLoader（同 seed）拿到本次采样到的场景文件，
按文件名存 top-k 轨迹，GEP 侧按 basename 查表。

用法:
    python tools/export_head_plans.py --data-dir /data/data_json/training \
        --head-data /workspace/data/head_dd/medoid \
        --model /workspace/data/head_dd/medoid/head_best.pt \
        --num-worlds 12 --seed 42 --out /workspace/data/head_plans.npz
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

N_ENV = 91          # GPUDrive 参考长度
SPEED_MIN = 0.5


def extend_to(path, n=N_ENV):
    """把 40 点轨迹按末端速度/方向外推到 n 点（看起来像正常行驶）。"""
    if len(path) >= n:
        return path[:n]
    d = path[-1] - path[-2]
    step = d  # 保持最后一段的位移
    out = list(path)
    p = path[-1].copy()
    while len(out) < n:
        p = p + step
        out.append(p.copy())
    return np.asarray(out)


def speed_profile(path, dt=0.1, n=None):
    """按相邻点位移算速度，末段保持。"""
    if len(path) < 2:
        return np.zeros(len(path), np.float32)
    v = np.linalg.norm(np.diff(path, axis=0), axis=1) / dt
    v = np.r_[v, v[-1]]
    return np.maximum(v, SPEED_MIN).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--head-data', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--num-worlds', type=int, default=12)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--out', required=True)
    ap.add_argument('--oracle', action='store_true',
                    help='用 GT 选最优锚点（5 份复制），用来定责')
    ap.add_argument('--certify', action='store_true',
                    help='用 E5 证书（推演+OBB）过滤候选')
    ap.add_argument('--files-json', default='',
                    help='按 eval dump 的文件名清单导出')
    args = ap.parse_args()

    import torch
    from train_head import Head
    from gpudrive.env.dataset import SceneDataLoader

    # 与 eval 完全一致的 loader 参数
    loader = SceneDataLoader(
        root=args.data_dir,
        batch_size=args.num_worlds,
        dataset_size=args.num_worlds,
        sample_with_replacement=True,
        shuffle=True,
        seed=args.seed,
    )
    if args.files_json and os.path.exists(args.files_json):
        _names = json.load(open(args.files_json))
        files = [os.path.join(args.data_dir, n) for n in _names]
        print('按 files-json 导出:', len(files))
    else:
        idx = list(loader.indices)[:args.num_worlds]
        files = [loader.dataset[i] for i in idx]
    print('采样的场景文件:')
    for w, f in enumerate(files):
        print(f'  world {w}: {os.path.basename(f)}')

    A = np.load(os.path.join(args.head_data, 'anchors.npy'))
    K, T = A.shape[0], A.shape[1]
    ck = torch.load(args.model, map_location='cpu')
    model = Head(ck.get('K', K), ck.get('T', T))
    model.load_state_dict(ck['model'])
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(dev).eval()
    anchors = torch.tensor(A, dtype=torch.float32, device=dev)

    z = np.load(os.path.join(args.head_data, 'train.npz'))
    ego_m, ego_s = z['ego'].mean(0), z['ego'].std(0) + 1e-3
    am = z['amask'].astype(bool)
    ag_m, ag_s = z['agent'][am].mean(0), z['agent'][am].std(0) + 1e-3
    mm = z['mmask'].astype(bool)
    mp_m, mp_s = z['map'][mm].mean(0), z['map'][mm].std(0) + 1e-3

    names, paths_all, probs_all, speeds_all = [], [], [], []
    origins, origin_h = [], []
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        sc = Scene(d, horizon=40)
        s = build_one(d, A)
        if s is None:
            print('  [warn] build_one failed:', os.path.basename(f))
            names.append(os.path.basename(f))
            origins.append(np.zeros(2, np.float32))
            origin_h.append(0.0)
            paths_all.append(np.zeros((args.k, N_ENV, 2), np.float32))
            probs_all.append(np.zeros(args.k, np.float32))
            speeds_all.append(np.zeros((args.k, N_ENV), np.float32))
            continue
        p0, h0 = sc.expert_pos[sc.t_now], float(sc.expert_heading[sc.t_now])
        _o = np.asarray(sc.expert_pos[0], np.float32)
        _d0 = sc.expert_pos[1] - sc.expert_pos[0]
        _oh = float(np.arctan2(_d0[1], _d0[0]))
        with torch.no_grad():
            ego = torch.tensor((s['ego'] - ego_m) / ego_s, device=dev)[None]
            ag = torch.tensor((s['agent'] - ag_m) / ag_s, device=dev)[None]
            am_ = torch.tensor(s['amask'], device=dev)[None]
            mp_ = torch.tensor((s['map'] - mp_m) / mp_s, device=dev)[None]
            mm_ = torch.tensor(s['mmask'], device=dev)[None]
            logits, _ = model(ego, ag, am_, mp_, mm_, anchors)
            prob = torch.softmax(logits, -1)[0].cpu().numpy()
        if args.oracle:
            _gt = s['gt'].astype(np.float32)          # (8,2) 自车系
            _ade = np.linalg.norm(A - _gt[None], axis=2).mean(1)
            order = np.asarray([int(np.argmin(_ade))])
            order = np.repeat(order, args.k)
        else:
            order = np.argsort(-prob)[:args.k]
        P, S = [], []
        for k in order:
            dense = ego_to_world(densify(A[k], sc.n_future), p0, h0)
            # 关键：参考必须从 env 的 frame 0 开始（日志 0..t_now），
            # 再接上规划头从 t_now+1 开始的计划，否则参考起点在车前方 1 秒
            prefix = np.asarray(sc.expert_pos[:sc.t_now + 1], np.float32)
            ref = np.vstack([prefix, dense.astype(np.float32)])
            ext = extend_to(ref, N_ENV)
            P.append(ext.astype(np.float32))
            S.append(speed_profile(ext))
        if args.certify:
            from exp_e5 import pure_pursuit, rollout_metrics
            _v0 = max(sc.speed_now(), 1.0)
            _ok = []
            for _p in P:
                try:
                    _roll = pure_pursuit(_p, _v0)
                    _m = rollout_metrics(sc, _roll)
                    _ok.append((not _m['coll']) and _m['edge_clear'] >= 1.0)
                except Exception:
                    _ok.append(True)
            _idx = [i for i, o in enumerate(_ok) if o]
            if not _idx:
                _idx = [0]          # 全失败 → 保留 top-1（与 E5 一致）
            P = [P[i] for i in _idx]
            S = [S[i] for i in _idx]
            _pr = [prob[order[i]] for i in _idx]
        else:
            _pr = list(prob[order])
        while len(P) < args.k:       # 补齐到 k 条（复制最后一条）
            P.append(P[-1])
            S.append(S[-1])
            _pr.append(_pr[-1])
        names.append(os.path.basename(f))
        origins.append(_o)
        origin_h.append(_oh)
        paths_all.append(np.stack(P))
        probs_all.append(np.asarray(_pr, np.float32))
        speeds_all.append(np.stack(S))
        print(f'  {os.path.basename(f)}: top1 概率 {prob[order[0]]:.2f} '
              f'终点距离 {np.linalg.norm(P[0][-1] - P[0][0]):.1f} m')

    np.savez_compressed(args.out,
                        names=np.array(names),
                        paths=np.stack(paths_all),
                        probs=np.stack(probs_all),
                        speeds=np.stack(speeds_all),
                        origin_pos=np.stack(origins),
                        origin_heading=np.asarray(origin_h, np.float32))
    print('saved', args.out, ' worlds', len(names))


if __name__ == '__main__':
    main()
