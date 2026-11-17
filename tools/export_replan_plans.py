"""导出滚动重规划用的多时刻规划（decision frames = 10, 25, 40, ...）。"""
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
from export_head_plans import extend_to, speed_profile  # noqa: E402

N_ENV = 91


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--head-data', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--files-json', required=True)
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--replan-every', type=int, default=15)
    ap.add_argument('--num-replans', type=int, default=6)
    ap.add_argument('--oracle', action='store_true',
                    help='用 GT 选最优锚点（上限实验）')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    import torch
    from train_head import Head

    names_in = json.load(open(args.files_json))
    files = [os.path.join(args.data_dir, n) for n in names_in]

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

    frames = [10 + args.replan_every * r for r in range(args.num_replans)]
    names, origins, origin_h = [], [], []
    P_all, S_all = [], []
    O_all, H_all = [], []

    for f in files:
        d = json.load(open(f))
        sc0 = Scene(d, horizon=40)
        _o = np.asarray(sc0.expert_pos[0], np.float32)
        _d0 = sc0.expert_pos[1] - sc0.expert_pos[0]
        _oh = float(np.arctan2(_d0[1], _d0[0]))
        P_r, S_r = [], []
        O_r, H_r = [], []
        for t in frames:
            if t >= sc0.T - 20:
                P_r.append(P_r[-1] if P_r else np.zeros((args.k, N_ENV, 2), np.float32))
                S_r.append(S_r[-1] if S_r else np.zeros((args.k, N_ENV), np.float32))
                O_r.append(O_r[-1] if O_r else np.zeros(2, np.float32))
                H_r.append(H_r[-1] if H_r else 0.0)
                continue
            sc = Scene(d, horizon=40, t_now=t)
            s = build_one(d, A, t_now=t)
            if s is None:                  # 视野不够（t+40 > 90）→ 沿用上一时刻
                P_r.append(P_r[-1] if P_r else np.zeros((args.k, N_ENV, 2), np.float32))
                S_r.append(S_r[-1] if S_r else np.zeros((args.k, N_ENV), np.float32))
                O_r.append(O_r[-1] if O_r else np.zeros(2, np.float32))
                H_r.append(H_r[-1] if H_r else 0.0)
                continue
            p0, h0 = sc.expert_pos[sc.t_now], float(sc.expert_heading[sc.t_now])
            O_r.append(np.asarray(p0, np.float32))
            H_r.append(float(h0))
            with torch.no_grad():
                ego = torch.tensor((s['ego'] - ego_m) / ego_s, device=dev)[None]
                ag = torch.tensor((s['agent'] - ag_m) / ag_s, device=dev)[None]
                am_ = torch.tensor(s['amask'], device=dev)[None]
                mp_ = torch.tensor((s['map'] - mp_m) / mp_s, device=dev)[None]
                mm_ = torch.tensor(s['mmask'], device=dev)[None]
                logits, _ = model(ego, ag, am_, mp_, mm_, anchors)
                prob = torch.softmax(logits, -1)[0].cpu().numpy()
            if args.oracle:
                _ade = np.linalg.norm(A - s['gt'][None], axis=2).mean(1)
                order = np.repeat(np.asarray([int(np.argmin(_ade))]), args.k)
            else:
                order = np.argsort(-prob)[:args.k]
            Pk, Sk = [], []
            for k in order:
                dense = ego_to_world(densify(A[k], sc.n_future), p0, h0)
                if t == frames[0]:
                    pre = np.asarray(sc.expert_pos[:t + 1], np.float32)
                    ref = np.vstack([pre, dense.astype(np.float32)])
                else:
                    ref = dense.astype(np.float32)
                ext = extend_to(ref, N_ENV)
                Pk.append(ext)
                Sk.append(speed_profile(ext))
            P_r.append(np.stack(Pk))
            S_r.append(np.stack(Sk))
        names.append(os.path.basename(f))
        origins.append(_o)
        origin_h.append(_oh)
        O_all.append(np.stack(O_r))
        H_all.append(np.asarray(H_r, np.float32))
        P_all.append(np.stack(P_r))
        S_all.append(np.stack(S_r))
        print('  ', os.path.basename(f), 'replans', len(P_r), flush=True)

    np.savez_compressed(args.out,
                        names=np.array(names),
                        frames=np.asarray(frames, np.int32),
                        paths=np.stack(P_all),      # (N, R, K, 91, 2)
                        speeds=np.stack(S_all),     # (N, R, K, 91)
                        origin_pos=np.stack(origins),
                        origin_heading=np.asarray(origin_h, np.float32),
                        plan_origin=np.stack(O_all),
                        plan_heading=np.stack(H_all))
    print('saved', args.out, ' worlds', len(names), ' frames', frames)


if __name__ == '__main__':
    main()
