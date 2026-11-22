"""A/B 对比评估：minADE/minFDE + 每个模态的 PDMS（含 oracle 上界）。

用法：
  python tools/eval_ab_pdms.py --data /workspace/data/planner60k \
      --ckpt /workspace/data/ab60k_B/latest.pt --split val --limit 200 \
      --out /workspace/data/ab60k_B/eval.npz
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, '/gpudrive/baselines/IDC-Bridge/src')
sys.path.insert(0, '/gpudrive/baselines/IDC-Bridge/tools')

from metrics.openloop_pdms import Scene, score_trajectory  # noqa: E402
from train_planner import Planner, densify, ego_to_world  # noqa: E402
from exp_pre import pure_pursuit  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/workspace/data/planner60k')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--split', default='val')
    ap.add_argument('--limit', type=int, default=200)
    ap.add_argument('--data-dir', default='/data/data_json/training')
    ap.add_argument('--v0', type=float, default=10.0)
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    z = np.load(f'{args.data}/{args.split}.npz', allow_pickle=True)
    N = min(len(z['ego']), args.limit)
    ck = torch.load(args.ckpt, map_location='cpu')
    K = ck.get('K', 6)
    model = Planner(K=K).to(dev).eval()
    model.load_state_dict(ck['model'], strict=False)

    preds = []
    with torch.no_grad():
        for b0 in range(0, N, 64):
            sl = slice(b0, min(b0 + 64, N))
            pred = model(torch.tensor(z['ego'][sl], dtype=torch.float32, device=dev),
                         torch.tensor(z['agent'][sl], dtype=torch.float32, device=dev),
                         torch.tensor(z['amask'][sl], dtype=torch.float32, device=dev),
                         torch.tensor(z['map'][sl], dtype=torch.float32, device=dev),
                         torch.tensor(z['mmask'][sl], dtype=torch.float32, device=dev))
            preds.append(pred.cpu().numpy())
    pred = np.concatenate(preds)                     # (N,K,8,2)

    gt = z['gt'][:N]
    d = np.linalg.norm(pred - gt[:, None], axis=-1)   # (N,K,8)
    ade = d.mean(-1)
    fde = d[:, :, -1]
    print(f'{args.split} 样本 {N}  K={K}  ckpt={os.path.basename(os.path.dirname(args.ckpt))}')
    print(f'minADE: p50={np.median(ade.min(1)):.3f} p90={np.percentile(ade.min(1), 90):.3f}')
    print(f'minFDE: p50={np.median(fde.min(1)):.3f} p90={np.percentile(fde.min(1), 90):.3f}'
          f' <2m={100*np.mean(fde.min(1) < 2):.0f}%')
    print(f'各模态 FDE<2m: {[round(float(x), 2) for x in (fde < 2.0).mean(0)]}')

    # ---- PDMS（执行器用纯追踪代理）----
    pdms = np.full((N, K), np.nan)
    for i in range(N):
        name = str(z['scene_id'][i])
        fp = os.path.join(args.data_dir, name)
        if not os.path.exists(fp):
            continue
        sc = Scene(json.load(open(fp)), horizon=40)
        ego_w = z['ego_world'][i]
        for k in range(K):
            c40 = densify(pred[i, k])
            cw = ego_to_world(c40, ego_w[:2], ego_w[2])
            try:
                roll = np.asarray(pure_pursuit(cw, args.v0), np.float64)
            except Exception:
                roll = cw
            H = sc.n_future
            if len(roll) < H:
                roll = np.vstack([roll, np.repeat(roll[-1:], H - len(roll), 0)])
            pdms[i, k] = score_trajectory(sc, roll[:H])['driving_score']

    ok = np.isfinite(pdms).all(1)
    print(f'PDMS（纯追踪执行）有效样本 {ok.sum()}/{N}')
    print('  各模态 PDMS 均值: %s' % [round(float(np.nanmean(pdms[ok, k])), 1) for k in range(K)])
    print('  各模态 PDMS 中位: %s' % [round(float(np.nanmedian(pdms[ok, k])), 1) for k in range(K)])
    best = np.nanmax(pdms[ok], 1)
    print(f'  oracle-over-K: 均值 {best.mean():.1f} 中位 {np.median(best):.1f} '
          f'p10 {np.percentile(best, 10):.1f}')
    kbest_fde = fde[ok].argmin(1)
    sel = pdms[ok][np.arange(ok.sum()), kbest_fde]
    print(f'  按 minFDE 选模态: 均值 {sel.mean():.1f} 中位 {np.median(sel):.1f}')
    print(f'  PDMS>0 的比例(全模态) {100*np.mean(pdms[ok] > 0):.0f}%  '
          f'>60 的比例 {100*np.mean(pdms[ok] > 60):.0f}%')
    if args.out:
        np.savez(args.out, pdms=pdms, ade=ade, fde=fde, ok=ok)
        print(f'  写出 {args.out}')


if __name__ == '__main__':
    main()
