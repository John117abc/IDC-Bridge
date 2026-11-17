"""规划头开环 PDMS 评测（支持不同预测时域）。

对比：expert / oracle(锚点最优) / const_vel / head
--horizon 40 → 只评前 4 秒（与 DiffusionDrive 对齐，8 waypoint × 0.5 s）
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "tools"))

from metrics.openloop_pdms import (Scene, _min_dist,  # noqa: E402
                                   score_trajectory)
from build_head_dataset import build_one, to_ego  # noqa: E402


def densify(wp, H):
    """8 waypoint(0.5s) -> H 个密集点(0.1s)，线性插值。

    关键：时间轴必须从 0 开始（起点在原点），否则前 0.5 s 原地不动，
    速度=0 → 方向判定失败 → PDMS 全 0。
    """
    tw = np.r_[0.0, np.arange(1, len(wp) + 1) * 0.5]
    wx = np.r_[0.0, wp[:, 0]]
    wy = np.r_[0.0, wp[:, 1]]
    td = np.arange(1, H + 1) * 0.1
    return np.stack([np.interp(td, tw, wx), np.interp(td, tw, wy)], 1)


def ego_to_world(pts, p0, h0):
    c, s = np.cos(h0), np.sin(h0)
    R = np.array([[c, -s], [s, c]])
    return pts @ R.T + np.asarray(p0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--n-scenes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--horizon", type=int, default=0, help="0=用完整未来窗口")
    ap.add_argument("--no-resid", action="store_true")
    ap.add_argument("--gate", type=int, default=0)
    args = ap.parse_args()

    import torch
    from train_head import Head

    anchors_np = np.load(os.path.join(args.data, 'anchors.npy'))
    K, T = anchors_np.shape[0], anchors_np.shape[1]
    ck = torch.load(args.model, map_location='cpu')
    model = Head(ck.get('K', K), ck.get('T', T))
    model.load_state_dict(ck['model'])
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(dev).eval()
    anchors = torch.tensor(anchors_np, dtype=torch.float32, device=dev)

    z = np.load(os.path.join(args.data, 'train.npz'))
    ego_m, ego_s = z['ego'].mean(0), z['ego'].std(0) + 1e-3
    am = z['amask'].astype(bool)
    ag_m, ag_s = z['agent'][am].mean(0), z['agent'][am].std(0) + 1e-3
    mm = z['mmask'].astype(bool)
    mp_m, mp_s = z['map'][mm].mean(0), z['map'][mm].std(0) + 1e-3

    files = sorted(glob.glob(os.path.join(args.data_dir, 'tfrecord-*.json')))
    rng = np.random.default_rng(args.seed)
    if len(files) > args.n_scenes:
        files = [files[i] for i in rng.choice(len(files), args.n_scenes,
                                              replace=False)]

    res = {k: [] for k in ('expert', 'oracle', 'const_vel', 'head')}
    fdes = []
    for fp in files:
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        sc = Scene(d, horizon=args.horizon if args.horizon > 0 else None)
        H = sc.n_future
        if H < 20:
            continue
        s = build_one(d, anchors_np)
        if s is None:
            continue
        p0, h0 = sc.expert_pos[sc.t_now], float(sc.expert_heading[sc.t_now])
        gt = sc.expert_future[:H]

        # 锚点覆盖率（在此时域下）
        gt_ego = to_ego(gt, p0, h0)
        best_fde = min(float(np.linalg.norm(
            densify(anchors_np[k], H)[-1] - gt_ego[-1])) for k in range(K))
        fdes.append(best_fde)

        r = score_trajectory(sc, gt)
        if r.get('valid'):
            res['expert'].append(r['driving_score'])

        best = 0.0
        for k in range(K):
            rk = score_trajectory(sc, ego_to_world(densify(anchors_np[k], H), p0, h0))
            if rk.get('valid'):
                best = max(best, rk['driving_score'])
        res['oracle'].append(best)

        v = max(sc.speed_now(), 0.5)
        t = np.arange(1, H + 1) * 0.1
        cv = np.stack([p0[0] + v * t * np.cos(h0),
                       p0[1] + v * t * np.sin(h0)], 1)
        r = score_trajectory(sc, cv)
        if r.get('valid'):
            res['const_vel'].append(r['driving_score'])

        with torch.no_grad():
            ego = torch.tensor((s['ego'] - ego_m) / ego_s, device=dev)[None]
            ag = torch.tensor((s['agent'] - ag_m) / ag_s, device=dev)[None]
            am_ = torch.tensor(s['amask'], device=dev)[None]
            mp_ = torch.tensor((s['map'] - mp_m) / mp_s, device=dev)[None]
            mm_ = torch.tensor(s['mmask'], device=dev)[None]
            logits, resid = model(ego, ag, am_, mp_, mm_, anchors)
            k = int(logits.argmax(-1))

        if args.gate > 0:
            top = torch.topk(logits, min(args.gate, K)).indices[0].tolist()
            best_k, best_clr = top[0], -1e9
            for kk in top:
                wt = ego_to_world(anchors_np[kk][:H], p0, h0)
                clr = (min(_min_dist(q, sc.ea, sc.eb) for q in wt)
                       if len(sc.ea) else 1e9)
                if clr > best_clr:
                    best_clr, best_k = clr, kk
            k = best_k

        pred_ego = (anchors_np[k] if args.no_resid
                    else (anchors[k] + resid[0]).cpu().numpy())
        r = score_trajectory(sc, ego_to_world(densify(pred_ego, H), p0, h0))
        if r.get('valid'):
            res['head'].append(r['driving_score'])

    f = np.asarray(fdes)
    print(f"\n场景 {len(res['expert'])}  时域 {H*0.1:.1f}s")
    print(f"锚点覆盖率: FDE<1m {100*np.mean(f<1):.1f}%  "
          f"p50={np.median(f):.2f}  p90={np.percentile(f,90):.2f}")
    print(f"{'方法':12s} {'PDMS 均值':>9s} {'p10':>6s} {'p50':>6s} {'p90':>6s}")
    for k in ('expert', 'oracle', 'head', 'const_vel'):
        a = np.asarray(res[k])
        if len(a):
            print(f"{k:12s} {a.mean():9.1f} {np.percentile(a,10):6.1f} "
                  f"{np.percentile(a,50):6.1f} {np.percentile(a,90):6.1f}")


if __name__ == '__main__':
    main()
