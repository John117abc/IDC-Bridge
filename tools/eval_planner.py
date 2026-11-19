"""上层规划器评估：minADE/minFDE、各模态质量、覆盖率。"""
import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, '/gpudrive/baselines/IDC-Bridge/src')
sys.path.insert(0, '/gpudrive/baselines/IDC-Bridge/tools')
from train_planner import Planner  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/workspace/data/planner5k')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--split', default='val')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    z = np.load(f'{args.data}/{args.split}.npz', allow_pickle=True)
    N = len(z['ego'])
    if args.limit:
        N = min(N, args.limit)
    ck = torch.load(args.ckpt, map_location='cpu')
    model = Planner(K=ck.get('K', 6)).to(dev).eval()
    model.load_state_dict(ck['model'])

    ade_all, fde_all = [], []
    with torch.no_grad():
        for b0 in range(0, N, 64):
            sl = slice(b0, min(b0 + 64, N))
            ego = torch.tensor(z['ego'][sl], dtype=torch.float32, device=dev)
            ag = torch.tensor(z['agent'][sl], dtype=torch.float32, device=dev)
            am = torch.tensor(z['amask'][sl], dtype=torch.float32, device=dev)
            mp = torch.tensor(z['map'][sl], dtype=torch.float32, device=dev)
            mm = torch.tensor(z['mmask'][sl], dtype=torch.float32, device=dev)
            gt = torch.tensor(z['gt'][sl], dtype=torch.float32, device=dev)
            pred = model(ego, ag, am, mp, mm)              # (B,K,8,2)
            d = (pred - gt[:, None]).norm(dim=-1)          # (B,K,8)
            ade = d.mean(-1)                               # (B,K)
            fde = d[:, :, -1]                              # (B,K)
            ade_all.append(ade.cpu().numpy())
            fde_all.append(fde.cpu().numpy())
    ade = np.concatenate(ade_all)
    fde = np.concatenate(fde_all)
    K = ade.shape[1]
    n = len(ade)
    mode_ok = (fde < 2.0).mean(0)          # 每个模态的"好"比例
    print(f'{args.split} 样本 {n}  K={K}')
    print(f'minADE: p50={np.median(ade.min(1)):.3f} '
          f'p90={np.percentile(ade.min(1), 90):.3f}')
    print(f'minFDE: p50={np.median(fde.min(1)):.3f} '
          f'p90={np.percentile(fde.min(1), 90):.3f} '
          f'<2m={100*np.mean(fde.min(1) < 2):.0f}%')
    print(f'各模态 FDE<2m 比例: '
          f'{[round(float(x), 2) for x in mode_ok]}')
    print(f'多模态冗余度（FDE<2m 的模态数分布）:',
          np.bincount((fde < 2.0).sum(1), minlength=K + 1).tolist())


if __name__ == '__main__':
    main()
