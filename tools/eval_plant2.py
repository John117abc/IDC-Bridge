"""M2 四项评估：精度 / 逐模态 / 多样性 / route-copy 基线。

说明：数据里没有未来周车轨迹，所以算不了真 PDMS；这里用可算的代理指标：
  - off-lane：轨迹点到最近车道中心线 >1.75 m 的比例
  - progress：轨迹末点沿 route 的弧长 / route 总长
"""
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


def lane_offset(traj, mp, mm):
    """轨迹点 (H,2) 到最近有效地图段的距离（返回每点最小距离）。"""
    d = np.full(len(traj), 10.0, np.float32)
    for k in range(len(mp)):
        if mm[k] < 0.5:
            continue
        mx, my, ln, c, s = mp[k, 0], mp[k, 1], mp[k, 2], mp[k, 3], mp[k, 4]
        dvec = np.array([c, s], np.float32) * ln / 2
        a = np.array([mx, my], np.float32) - dvec
        b = np.array([mx, my], np.float32) + dvec
        ab = b - a
        t = np.clip(((traj - a) @ ab) / max(float(ab @ ab), 1e-9), 0, 1)
        proj = a + t[:, None] * ab
        d = np.minimum(d, np.linalg.norm(traj - proj, axis=1))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/mnt/2T_HDD/plant2_vector/vector_ds_full')
    ap.add_argument('--ckpt', default='/mnt/2T_HDD/plant2_vector/ckpt/best.pt')
    ap.add_argument('--limit', type=int, default=3000)
    ap.add_argument('--out', default='/mnt/2T_HDD/plant2_vector/eval_report.json')
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.data, '*.npz')))
    E, A, AM, M, MM, R, G = load(paths)
    MP, MM2 = add_route_tokens(M, MM, R)
    del M, MM
    n = len(G)
    idx = np.random.RandomState(0).permutation(n)[:args.limit]
    E, A, AM, MP, MM2, R, G = (E[idx], A[idx], AM[idx], MP[idx], MM2[idx], R[idx], G[idx])
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ck = torch.load(args.ckpt, map_location='cpu')
    K = int(ck.get('K', 6))
    model = Planner(K=K).to(dev)
    model.load_state_dict(ck['model'])
    model.eval()
    print(f'评估 {len(G)} 样本, K={K}', flush=True)

    preds = []
    with torch.no_grad():
        for i in range(0, len(G), 256):
            b = [torch.tensor(x[i:i + 256], device=dev) for x in (E, A, AM, MP, MM2)]
            preds.append(model(*b).cpu().numpy())
    P = np.concatenate(preds)
    err = np.abs(P - G[:, None]).mean(-1)          # (N,K,8)
    ade_k = err.mean(-1)                            # (N,K)
    fde_k = err[..., -1]
    speed = np.linalg.norm(G[:, 1] - G[:, 0], axis=1) / 0.5   # 由 GT 估速度

    # route-copy 基线：沿 route 以 GT 平均速度铺 8 点
    rc = np.zeros_like(G)
    for i in range(len(G)):
        s = R[i, :, 2]
        ok = np.abs(R[i, :, 0]) + np.abs(R[i, :, 1]) > 0
        sp = speed[i] * np.arange(1, 9) * 0.5
        if ok.sum() >= 2:
            sx, sy = s[ok], R[i, ok, :2]
            rc[i, :, 0] = np.interp(sp, sx, sx * 0 + np.arange(len(sx)))
            rc[i, :, 0] = np.interp(sp, sx, sy[:, 0])
            rc[i, :, 1] = np.interp(sp, sx, sy[:, 1])
    err_rc = np.abs(rc - G).mean(-1)
    off = []
    for i in range(0, len(G), 50):
        d = lane_offset(P[i, 0], MP[i, :128], MM2[i, :128])
        off.append(float((d > 1.75).mean()))
    div = np.linalg.norm(P[:, :, -1, None, :] - P[:, None, :, -1, :], axis=-1)
    kw = ~np.eye(K, dtype=bool)[None]
    minpair = np.where(np.broadcast_to(kw, div.shape), div, np.inf).min(axis=(1, 2))

    rep = dict(
        n=int(len(G)), K=K,
        minADE=float(ade_k.min(1).mean()), minFDE=float(fde_k.min(1).mean()),
        top1_ADE=float(ade_k[:, 0].mean()), top1_FDE=float(fde_k[:, 0].mean()),
        oracle_FDE=float(fde_k.min(1).mean()),
        per_mode_FDE=[float(x) for x in fde_k.mean(0)],
        per_mode_winrate=[float(x) for x in (fde_k.argmin(1)[:, None] == np.arange(K)[None]).mean(0)],
        div_minpair_med=float(np.median(minpair)),
        div_all_gt2m=float((minpair > 2).mean()),
        dup_rate=float((minpair < 0.5).mean()),
        offlane_frac=float(np.mean(off)),
        routecopy_ADE=float(err_rc.mean()), routecopy_FDE=float(err_rc[:, -1].mean()),
    )
    for k, v in rep.items():
        print(f'{k}: {v}', flush=True)
    json.dump(rep, open(args.out, 'w'), indent=2, ensure_ascii=False)
    print('saved', args.out)


if __name__ == '__main__':
    main()
