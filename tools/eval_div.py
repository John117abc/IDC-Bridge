"""评估模态多样性 + 轨迹精度（单个 checkpoint）。"""
import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, '/gpudrive/baselines/IDC-Bridge/src')
sys.path.insert(0, '/gpudrive/baselines/IDC-Bridge/tools')
from train_planner import Planner  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/workspace/data/planner60k')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--split', default='val')
    ap.add_argument('--limit', type=int, default=200)
    ap.add_argument('--tag', default='')
    ap.add_argument('--pdms-npz', default='',
                    help='可选：eval_ab_pdms 输出的 npz，用于统计"互不相同且可用"的模态数')
    args = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    z = np.load(f'{args.data}/{args.split}.npz', allow_pickle=True)
    n = min(len(z['ego']), args.limit)
    ck = torch.load(args.ckpt, map_location='cpu')
    K = ck.get('K', 6)
    m = Planner(K=K).to(dev).eval()
    m.load_state_dict(ck['model'], strict=False)
    out = []
    with torch.no_grad():
        for b0 in range(0, n, 64):
            sl = slice(b0, min(b0 + 64, n))
            out.append(m(torch.tensor(z['ego'][sl], dtype=torch.float32, device=dev),
                         torch.tensor(z['agent'][sl], dtype=torch.float32, device=dev),
                         torch.tensor(z['amask'][sl], dtype=torch.float32, device=dev),
                         torch.tensor(z['map'][sl], dtype=torch.float32, device=dev),
                         torch.tensor(z['mmask'][sl], dtype=torch.float32, device=dev)
                         ).cpu().numpy())
    p = np.concatenate(out)
    gt = z['gt'][:n]
    iu = np.triu_indices(K, 1)
    stop = np.linalg.norm(p[:, iu[0], -1] - p[:, iu[1], -1], axis=-1)   # (n,P)
    mid = np.linalg.norm(p[:, iu[0], 4] - p[:, iu[1], 4], axis=-1)
    fde = np.linalg.norm(p - gt[:, None], axis=-1)[:, :, -1]
    ade = np.linalg.norm(p - gt[:, None], axis=-1).mean(-1)
    tag = args.tag or args.ckpt.split('/')[-2]
    print('== %s (n=%d, K=%d) ==' % (tag, n, K))
    print('  终点两两距离: 中位 %.2f m | 每场景最小对 中位 %.2f m'
          % (np.median(stop), np.median(stop.min(1))))
    print('  中点两两距离: 中位 %.2f m' % np.median(mid))
    print('  场景"全部两两 >2 m": %.0f%% | "有对 <1 m": %.0f%% | 完全重复对: %.1f%%'
          % (100 * np.mean(stop.min(1) > 2), 100 * np.mean(stop.min(1) < 1),
             100 * np.mean(stop < 1e-3)))
    print('  minADE p50 %.2f | minFDE p50 %.2f p90 %.2f | <2m %.0f%%'
          % (np.median(ade.min(1)), np.median(fde.min(1)),
             np.percentile(fde.min(1), 90), 100 * np.mean(fde.min(1) < 2)))
    print('  FDE<2m 模态数分布: %s'
          % np.bincount((fde < 2).sum(1), minlength=K + 1).tolist())

    # ---- "互不相同且可用"的模态数：贪心选 PDMS>0（或 FDE<2m）且两两终点 >2 m ----
    pair_idx = {(int(iu[0][t]), int(iu[1][t])): t
                for t in range(len(iu[0]))}

    def distinct_ok(valid):
        cnt = np.zeros(n, int)
        for i in range(n):
            chosen = []
            for k in np.argsort(-valid[i].astype(np.float32)):
                if not valid[i][k]:
                    continue
                okc = True
                for j in chosen:
                    a, b = (k, j) if k < j else (j, k)
                    if stop[i, pair_idx[(a, b)]] < 2.0:
                        okc = False
                        break
                if okc:
                    chosen.append(int(k))
            cnt[i] = len(chosen)
        return cnt

    cnt_fde = distinct_ok(fde < 2.0)
    print('  「互不相同且 FDE<2m」模态数: 均值 %.2f，分布 %s'
          % (cnt_fde.mean(), np.bincount(cnt_fde, minlength=K + 1).tolist()))
    if args.pdms_npz:
        pm = np.load(args.pdms_npz)['pdms'][:n]
        ok = np.isfinite(pm).all(1)
        cnt_p = distinct_ok(pm > 0)
        print('  「互不相同且 PDMS>0」模态数: 均值 %.2f，分布 %s'
              % (cnt_p[ok].mean(), np.bincount(cnt_p[ok], minlength=K + 1).tolist()))


if __name__ == '__main__':
    main()
