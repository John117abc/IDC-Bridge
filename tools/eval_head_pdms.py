"""用训练好的规划头跑开环 PDMS，并给出三条基线。

对比：
    expert     专家轨迹（=100，构造上限）
    oracle     16 条锚点里逐场景选最好（上界）
    const_vel  当前速度直行（最弱基线）
    head       规划头输出

用法：
    python tools/eval_head_pdms.py --data /workspace/data/head20 \
        --model /workspace/data/head20/head.pt \
        --data-dir /data/data_json/training --n-scenes 300
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

from metrics.openloop_pdms import Scene, score_trajectory  # noqa: E402
from build_head_dataset import build_one, kmeans, to_ego  # noqa: E402,F401


def ego_to_world(pts, p0, h0):
    c, s = np.cos(h0), np.sin(h0)
    R = np.array([[c, -s], [s, c]])
    return pts @ R.T + np.asarray(p0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help='含 anchors.npy 的目录')
    ap.add_argument("--model", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--n-scenes", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
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

    files = sorted(glob.glob(os.path.join(args.data_dir, 'tfrecord-*.json')))
    rng = np.random.default_rng(args.seed)
    if len(files) > args.n_scenes:
        files = [files[i] for i in rng.choice(len(files), args.n_scenes,
                                              replace=False)]

    # 归一化统计（与训练一致，直接从 npz 读）
    z = np.load(os.path.join(args.data, 'train.npz'))
    ego_m, ego_s = z['ego'].mean(0), z['ego'].std(0) + 1e-3
    am = z['amask'].astype(bool)
    ag_m, ag_s = z['agent'][am].mean(0), z['agent'][am].std(0) + 1e-3
    mm = z['mmask'].astype(bool)
    mp_m, mp_s = z['map'][mm].mean(0), z['map'][mm].std(0) + 1e-3

    res = {k: [] for k in ('expert', 'oracle', 'const_vel', 'head')}
    n = 0
    for fp in files:
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        sc = Scene(d)
        if sc.n_future < 20:
            continue
        s = build_one(d, anchors_np)
        if s is None:
            continue
        p0, h0 = sc.expert_pos[sc.t_now], float(sc.expert_heading[sc.t_now])
        n += 1

        # --- 专家 ---
        r = score_trajectory(sc, sc.expert_future)
        if r.get('valid'):
            res['expert'].append(r['driving_score'])

        # --- oracle：锚点 → 世界系 ---
        best = 0.0
        const = None
        for k in range(K):
            w = ego_to_world(anchors_np[k], p0, h0)
            rk = score_trajectory(sc, w)
            if rk.get('valid'):
                best = max(best, rk['driving_score'])
        res['oracle'].append(best)

        # --- 恒定速度直行 ---
        v = max(sc.speed_now(), 0.5)
        t = np.arange(1, sc.n_future + 1) * 0.1
        cv = np.stack([p0[0] + v * t * np.cos(h0),
                       p0[1] + v * t * np.sin(h0)], 1)
        r = score_trajectory(sc, cv)
        if r.get('valid'):
            res['const_vel'].append(r['driving_score'])

        # --- 规划头 ---
        with torch.no_grad():
            ego = torch.tensor((s['ego'] - ego_m) / ego_s, device=dev)[None]
            ag = torch.tensor((s['agent'] - ag_m) / ag_s, device=dev)[None]
            am_ = torch.tensor(s['amask'], device=dev)[None]
            mp_ = torch.tensor((s['map'] - mp_m) / mp_s, device=dev)[None]
            mm_ = torch.tensor(s['mmask'], device=dev)[None]
            logits, resid = model(ego, ag, am_, mp_, mm_, anchors)
            k = int(logits.argmax(-1))
            pred_ego = (anchors[k] + resid[0]).cpu().numpy()
        w = ego_to_world(pred_ego, p0, h0)
        r = score_trajectory(sc, w)
        if r.get('valid'):
            res['head'].append(r['driving_score'])

    print(f"\n场景 {n}")
    print(f"{'方法':12s} {'PDMS 均值':>9s} {'p10':>6s} {'p50':>6s} {'p90':>6s}")
    for k in ('expert', 'oracle', 'head', 'const_vel'):
        a = np.asarray(res[k])
        if len(a):
            print(f"{k:12s} {a.mean():9.1f} {np.percentile(a,10):6.1f} "
                  f"{np.percentile(a,50):6.1f} {np.percentile(a,90):6.1f}")


if __name__ == '__main__':
    main()
