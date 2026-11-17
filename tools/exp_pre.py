"""两个先导实验：
A. 规划噪声注入曲线 —— 给规划头输出加横向/纵向扰动，看开环 PDMS 衰减
B. 离线跟踪实验 —— 用纯追踪 / PID 去跟踪**规划头的轨迹**，量跟踪误差(gap)与 PDMS

用法: python tools/exp_pre.py --data /workspace/data/head_dd/medoid --model .../head_best.pt --n-scenes 120
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
from build_head_dataset import build_one  # noqa: E402
from eval_head_pdms import densify, ego_to_world  # noqa: E402

DT = 0.1
WHEELBASE = 2.8


def smoothstep5(s):
    s = np.clip(s, 0.0, 1.0)
    return s ** 3 * (s * (s * 6 - 15) + 10)


def lateral_offset(path, offset):
    """沿法向施加平滑横向偏移（两端为 0）。"""
    d = np.gradient(path, axis=0)
    n = np.stack([-d[:, 1], d[:, 0]], 1)
    nn = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.maximum(nn, 1e-6)
    w = smoothstep5(np.linspace(0, 1, len(path)))
    return path + offset * w[:, None] * n


def resample(path, scale):
    """按弧长缩放（纵向 x0.8 / x1.2）。"""
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    arc = np.r_[0, np.cumsum(seg)]
    total = arc[-1] * scale
    tgt = np.linspace(0, total, len(path))
    src = arc / max(arc[-1], 1e-6) * total
    return np.stack([np.interp(tgt, src, path[:, 0]),
                     np.interp(tgt, src, path[:, 1])], 1)


def pure_pursuit(ref, v0, lookahead=1.2, max_steer=0.6):
    """运动学自行车 + 纯追踪。ref: (T,2) 目标路径（含起点）。"""
    seg = np.linalg.norm(np.diff(ref, axis=0), axis=1)
    arc = np.r_[0, np.cumsum(seg)]
    p = ref[0].copy()
    h = np.arctan2(ref[1, 1] - ref[0, 1], ref[1, 0] - ref[0, 0])
    out = [p.copy()]
    v = v0
    for _ in range(len(ref) - 1):
        Ld = max(2.0, v * lookahead)
        j = int(np.searchsorted(arc, arc[min(len(arc) - 1, len(out))] + Ld))
        j = min(j, len(ref) - 1)
        tgt = ref[j]
        alpha = np.arctan2(tgt[1] - p[1], tgt[0] - p[0]) - h
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))
        delta = np.arctan2(2 * WHEELBASE * np.sin(alpha), max(Ld, 1e-3))
        delta = float(np.clip(delta, -max_steer, max_steer))
        # 参考速度
        k = min(len(ref) - 2, int(arc[min(len(arc) - 1, len(out))] / max(DT, 1e-6)))
        v_ref = np.linalg.norm(ref[min(k + 1, len(ref) - 1)] - ref[k]) / DT
        v = 0.7 * v + 0.3 * max(v_ref, 0.0)
        p = p + v * DT * np.array([np.cos(h), np.sin(h)])
        h = h + v * DT / WHEELBASE * np.tan(delta)
        out.append(p.copy())
    return np.asarray(out)


def stanley(ref, v0, k_cross=2.0, max_steer=0.6):
    """Stanley 控制器（横向误差 + 航向误差）。"""
    seg = np.linalg.norm(np.diff(ref, axis=0), axis=1)
    arc = np.r_[0, np.cumsum(seg)]
    p = ref[0].copy()
    h = np.arctan2(ref[1, 1] - ref[0, 1], ref[1, 0] - ref[0, 0])
    out = [p.copy()]
    v = v0
    for i in range(1, len(ref) - 1):
        # 最近点
        d = np.linalg.norm(ref - p[None], axis=1)
        j = int(np.argmin(d))
        j = min(j + 1, len(ref) - 1)
        tang = ref[j] - ref[j - 1]
        th = np.arctan2(tang[1], tang[0])
        n = np.array([-np.sin(th), np.cos(th)])
        cross = float((p - ref[j]) @ n)
        psi = np.arctan2(np.sin(th - h), np.cos(th - h))
        delta = psi + np.arctan2(-k_cross * cross, max(v, 1.0))
        delta = float(np.clip(delta, -max_steer, max_steer))
        v_ref = np.linalg.norm(ref[j] - ref[j - 1]) / DT
        v = 0.7 * v + 0.3 * max(v_ref, 0.0)
        p = p + v * DT * np.array([np.cos(h), np.sin(h)])
        h = h + v * DT / WHEELBASE * np.tan(delta)
        out.append(p.copy())
    return np.asarray(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--n-scenes", type=int, default=120)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    import torch
    from train_head import Head

    A = np.load(os.path.join(args.data, 'anchors.npy'))
    K, T = A.shape[0], A.shape[1]
    ck = torch.load(args.model, map_location='cpu')
    model = Head(ck.get('K', K), ck.get('T', T))
    model.load_state_dict(ck['model'])
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(dev).eval()
    anchors = torch.tensor(A, dtype=torch.float32, device=dev)

    z = np.load(os.path.join(args.data, 'train.npz'))
    ego_m, ego_s = z['ego'].mean(0), z['ego'].std(0) + 1e-3
    am = z['amask'].astype(bool)
    ag_m, ag_s = z['agent'][am].mean(0), z['agent'][am].std(0) + 1e-3
    mm = z['mmask'].astype(bool)
    mp_m, mp_s = z['map'][mm].mean(0), z['map'][mm].std(0) + 1e-3

    files = sorted(glob.glob(os.path.join(args.data_dir, 'tfrecord-*.json')))
    rng = np.random.default_rng(args.seed)
    files = [files[i] for i in rng.choice(len(files), args.n_scenes, replace=False)]

    # ---- A: 噪声/缩放 ----
    A_cases = {
        'nominal': lambda p: p,
        'lat +0.5': lambda p: lateral_offset(p, 0.5),
        'lat -0.5': lambda p: lateral_offset(p, -0.5),
        'lat +1.0': lambda p: lateral_offset(p, 1.0),
        'lat -1.0': lambda p: lateral_offset(p, -1.0),
        'lat +2.0': lambda p: lateral_offset(p, 2.0),
        'lat -2.0': lambda p: lateral_offset(p, -2.0),
        'long x0.8': lambda p: resample(p, 0.8),
        'long x1.2': lambda p: resample(p, 1.2),
    }
    a_res = {k: [] for k in A_cases}

    # ---- B: 跟踪 ----
    b_res = {k: {'gap': [], 'pdms': []}
             for k in ('pure_pursuit', 'stanley')}

    for fp in files:
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        sc = Scene(d, horizon=40)
        s = build_one(d, A)
        if s is None or sc.n_future < 20:
            continue
        H = sc.n_future
        p0, h0 = sc.expert_pos[sc.t_now], float(sc.expert_heading[sc.t_now])
        with torch.no_grad():
            ego = torch.tensor((s['ego'] - ego_m) / ego_s, device=dev)[None]
            ag = torch.tensor((s['agent'] - ag_m) / ag_s, device=dev)[None]
            am_ = torch.tensor(s['amask'], device=dev)[None]
            mp_ = torch.tensor((s['map'] - mp_m) / mp_s, device=dev)[None]
            mm_ = torch.tensor(s['mmask'], device=dev)[None]
            logits, _ = model(ego, ag, am_, mp_, mm_, anchors)
            k = int(logits.argmax(-1))
        ref = ego_to_world(densify(A[k], H), p0, h0)

        for name, fn in A_cases.items():
            r = score_trajectory(sc, fn(ref))
            if r.get('valid'):
                a_res[name].append(r['driving_score'])

        v0 = max(sc.speed_now(), 1.0)
        for name, fn in (('pure_pursuit', pure_pursuit), ('stanley', stanley)):
            try:
                exe = fn(ref, v0)
            except Exception:
                continue
            if len(exe) < H:
                exe = np.vstack([exe, np.repeat(exe[-1:], H - len(exe), 0)])
            n = min(len(exe), H)
            gap = float(np.mean(np.linalg.norm(exe[:n] - ref[:n], axis=1)))
            r = score_trajectory(sc, exe[:n])
            if r.get('valid'):
                b_res[name]['gap'].append(gap)
                b_res[name]['pdms'].append(r['driving_score'])

    print(f"\n=== A. 规划噪声注入（{len(a_res['nominal'])} 场景）===")
    print(f"{'变体':12s} {'PDMS 均值':>9s} {'p10':>6s} {'p50':>6s}   Δ均值")
    base = np.mean(a_res['nominal'])
    for name in A_cases:
        a = np.asarray(a_res[name])
        if len(a):
            print(f"{name:12s} {a.mean():9.1f} {np.percentile(a,10):6.1f} "
                  f"{np.percentile(a,50):6.1f}   {a.mean()-base:+.1f}")

    print(f"\n=== B. 跟踪规划轨迹（{len(b_res['stanley']['gap'])} 场景）===")
    print(f"{'控制器':14s} {'gap 均值(m)':>11s} {'gap p90':>8s} "
          f"{'PDMS 均值':>9s} {'p10':>6s}")
    for name in ('pure_pursuit', 'stanley'):
        g = np.asarray(b_res[name]['gap'])
        p = np.asarray(b_res[name]['pdms'])
        if len(g):
            print(f"{name:14s} {g.mean():11.3f} {np.percentile(g,90):8.3f} "
                  f"{p.mean():9.1f} {np.percentile(p,10):6.1f}")


if __name__ == '__main__':
    main()
