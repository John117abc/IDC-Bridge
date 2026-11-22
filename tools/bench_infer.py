"""量一下"给场景 → 出 K 条模态轨迹"这条链路的耗时与模态多样性。

三段：
  A. 前向推理延迟（GPU/CPU，batch 1/8/32）→ 能跑到多少 Hz
  B. 特征构建耗时（json 读取 / 特征 / 奖励几何三项拆开）
  C. 模态多样性（同一场景 K 条模态互相差多少）
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, '/gpudrive/baselines/IDC-Bridge/src')
sys.path.insert(0, '/gpudrive/baselines/IDC-Bridge/tools')

from train_planner import Planner  # noqa: E402
import build_head_dataset as B  # noqa: E402


def bench(fn, n=50, warmup=10):
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return 1000.0 * (time.time() - t0) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/workspace/data/planner60k')
    ap.add_argument('--ckpt', default='/workspace/data/ab60k_B/latest.pt')
    ap.add_argument('--data-dir', default='/data/data_json/training')
    ap.add_argument('--n-feat', type=int, default=20)
    ap.add_argument('--n-div', type=int, default=200)
    args = ap.parse_args()

    z = np.load(f'{args.data}/val.npz', allow_pickle=True)
    ck = torch.load(args.ckpt, map_location='cpu')
    K = ck.get('K', 6)

    # ---------- A. 前向延迟 ----------
    print('== A. 前向推理延迟（K=%d, 8 waypoint）==' % K)
    for dev in (['cuda', 'cpu'] if torch.cuda.is_available() else ['cpu']):
        m = Planner(K=K).to(dev).eval()
        m.load_state_dict(ck['model'])
        for bs in (1, 8, 32):
            ego = torch.randn(bs, 23, device=dev)
            ag = torch.randn(bs, 32, 28, device=dev)
            am = torch.ones(bs, 32, device=dev)
            mp = torch.randn(bs, 128, 7, device=dev)
            mm = torch.ones(bs, 128, device=dev)
            with torch.no_grad():
                f = lambda: m(ego, ag, am, mp, mm)  # noqa: E731
                ms = bench(f, n=30 if bs <= 8 else 15)
            print('  %-4s bs=%-2d  %.2f ms/次  → %.0f Hz（单样本等效 %.2f ms）'
                  % (dev, bs, ms, 1000.0 / ms, ms / bs))
        del m

    # ---------- B. 特征构建 ----------
    print('== B. 单场景特征构建（%d 场景平均）==' % args.n_feat)
    import glob
    files = sorted(glob.glob(os.path.join(args.data_dir, 'tfrecord-*.json')))
    rng = np.random.default_rng(0)
    files = [files[i] for i in rng.choice(len(files), args.n_feat, replace=False)]
    anc = np.zeros((20, 8, 2), np.float32)

    t_json = t_feat = t_full = 0.0
    for fp in files:
        t0 = time.time()
        d = json.load(open(fp))
        t_json += time.time() - t0
        t0 = time.time()
        B.build_one(d, anc)                       # 含奖励几何（Scene + step_metrics）
        t_full += time.time() - t0
    t_full /= len(files)
    t_json /= len(files)

    import metrics.openloop_pdms as pdms
    _Scene, _step = pdms.Scene, pdms.step_metrics
    pdms.Scene = None                             # 让 build_one 里 try 块抛错 → 跳过奖励
    for fp in files:
        d = json.load(open(fp))
        t0 = time.time()
        B.build_one(d, anc)
        t_feat += time.time() - t0
    pdms.Scene = _Scene
    t_feat /= len(files)
    print('  json 读取            %6.1f ms' % (1000 * t_json))
    print('  特征构建（不含奖励）  %6.1f ms' % (1000 * (t_feat - t_json)))
    print('  奖励几何（hm_* 那段）%6.1f ms' % (1000 * (t_full - t_feat)))
    print('  合计（离线口径）      %6.1f ms  → %.1f Hz'
          % (1000 * t_full, 1.0 / t_full))
    print('  合计（在线只需特征）  %6.1f ms  → %.1f Hz'
          % (1000 * t_feat, 1.0 / t_feat))

    # ---------- C. 模态多样性 ----------
    print('== C. 模态多样性（%d 场景 × K=%d）==' % (args.n_div, K))
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = Planner(K=K).to(dev).eval()
    m.load_state_dict(ck['model'])
    n = min(len(z['ego']), args.n_div)
    preds = []
    with torch.no_grad():
        for b0 in range(0, n, 64):
            sl = slice(b0, min(b0 + 64, n))
            preds.append(m(torch.tensor(z['ego'][sl], dtype=torch.float32, device=dev),
                           torch.tensor(z['agent'][sl], dtype=torch.float32, device=dev),
                           torch.tensor(z['amask'][sl], dtype=torch.float32, device=dev),
                           torch.tensor(z['map'][sl], dtype=torch.float32, device=dev),
                           torch.tensor(z['mmask'][sl], dtype=torch.float32, device=dev)
                           ).cpu().numpy())
    p = np.concatenate(preds)                    # (n,K,8,2)
    gt = z['gt'][:n]
    # 两两 FDE（终点距离）与整条轨迹差
    pair_fde, pair_ade = [], []
    for i in range(K):
        for j in range(i + 1, K):
            pair_fde.append(np.linalg.norm(p[:, i, -1] - p[:, j, -1], axis=-1))
            pair_ade.append(np.linalg.norm(p[:, i] - p[:, j], axis=-1).mean(-1))
    pf = np.stack(pair_fde, 1)                   # (n, K(K-1)/2)
    pa = np.stack(pair_ade, 1)
    print('  两两终点距离 min/max: %.2f / %.2f m（中位 %.2f）'
          % (np.median(pf.min(1)), np.median(pf.max(1)), np.median(pf)))
    print('  两两整条轨迹平均差: %.2f m（中位）' % np.median(pa))
    print('  场景中"至少有 2 条模态终点差 <1 m"的比例: %.0f%%'
          % (100 * np.mean(pf.min(1) < 1)))
    print('  场景中"6 条模态终点两两差 >2 m"的比例: %.0f%%'
          % (100 * np.mean(pf.min(1) > 2)))
    fde = np.linalg.norm(p - gt[:, None], axis=-1)[:, :, -1]
    good = (fde < 2.0)
    print('  每个场景 FDE<2m 的模态数分布: %s'
          % np.bincount(good.sum(1), minlength=K + 1).tolist())
    # 模态是否塌缩：同场景内两条模态完全相同（<1e-3 m）
    dup = (pa < 1e-3)
    print('  完全重复的模态对比例: %.1f%%' % (100 * dup.mean()))


if __name__ == '__main__':
    main()
