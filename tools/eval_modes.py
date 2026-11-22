"""逐模态诊断：条件 FDE（只在它赢的样本上算）+ win 分布熵 + 头部上界。"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, '/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/tools')
from train_plant2_small import load, add_route_tokens   # noqa: E402
from train_planner import Planner                      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/mnt/2T_HDD/plant2_vector/vector_ds_full')
    ap.add_argument('--ckpt', default='/mnt/2T_HDD/plant2_vector/ckpt/best.pt')
    ap.add_argument('--limit', type=int, default=5000)
    ap.add_argument('--out', default='/mnt/2T_HDD/plant2_vector/mode_diag.json')
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.data, '*.npz')))
    E, A, AM, M, MM, R, G = load(paths)
    MP, MM2 = add_route_tokens(M, MM, R)
    del M, MM
    idx = np.random.RandomState(0).permutation(len(G))[:args.limit]
    E, A, AM, MP, MM2, G = E[idx], A[idx], AM[idx], MP[idx], MM2[idx], G[idx]
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ck = torch.load(args.ckpt, map_location='cpu')
    K = int(ck.get('K', 6))
    model = Planner(K=K).to(dev)
    model.load_state_dict(ck['model'])
    model.eval()
    P = []
    with torch.no_grad():
        for i in range(0, len(G), 256):
            b = [torch.tensor(x[i:i + 256], device=dev) for x in (E, A, AM, MP, MM2)]
            P.append(model(*b).cpu().numpy())
    P = np.concatenate(P)
    fde = np.abs(P - G[:, None]).mean(-1)[..., -1]      # (N,K)
    win = fde.argmin(1)

    rows = []
    for k in range(K):
        m = win == k
        rows.append(dict(mode=k, winrate=float(m.mean()),
                         cond_FDE=float(fde[m, k].mean()) if m.any() else float('nan'),
                         always_FDE=float(fde[:, k].mean()),
                         p90_cond=float(np.percentile(fde[m, k], 90)) if m.any() else float('nan')))
    hist = np.bincount(win, minlength=K) / len(win)
    ent = float(-(hist * np.log(hist + 1e-9)).sum() / np.log(K))

    # 头部上界：完美的“赢家分类器” = 永远选对 → FDE = minFDE
    # 随机选 → 每模平均
    rep = dict(n=int(len(G)), K=K,
               minFDE=float(fde.min(1).mean()),
               conditional=[r for r in rows],
               win_hist=[float(x) for x in hist],
               win_entropy_norm=ent,
               random_pick_FDE=float(fde.mean()),
               best_single_mode=min(rows, key=lambda r: r['always_FDE']))
    for r in rows:
        print(f"mode{r['mode']}: win {r['winrate']*100:5.1f}%  "
              f"condFDE {r['cond_FDE']:.3f}  p90 {r['p90_cond']:.2f}  "
              f"alwaysFDE {r['always_FDE']:.3f}", flush=True)
    print(f"win 熵(归一化) {ent:.3f}   (1.0=完全均匀, 0=单模通吃)", flush=True)
    print(f"oracle(=minFDE) {rep['minFDE']:.3f}  随机选 {rep['random_pick_FDE']:.3f}  "
          f"最好的单模 {rep['best_single_mode']['mode']} "
          f"({rep['best_single_mode']['always_FDE']:.3f})", flush=True)
    json.dump(rep, open(args.out, 'w'), indent=2, ensure_ascii=False)
    print('saved', args.out)


if __name__ == '__main__':
    main()
