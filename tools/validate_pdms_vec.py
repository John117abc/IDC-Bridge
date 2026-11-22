"""检查点 3：向量化 PDMS 对齐验证。

两段验证：
  A. 实现对齐：同一份 Scene 几何下，score_batch(geo_from_scene) 应逐位等于
     score_trajectory（误差 < tol）。
  B. 数据口径保真度：数据集里的紧凑几何（geo_from_npz，最近 32 周车 +
     沿专家走廊 512 条线段）算出的 PDMS 与全量几何的差异。

用法：
  python tools/validate_pdms_vec.py --data /workspace/data/planner300/train.npz \
      --n-scenes 300 --g 4 --sigma 0.5
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, '/gpudrive/baselines/IDC-Bridge/src')

from metrics.openloop_pdms import Scene, score_trajectory  # noqa: E402
from metrics.openloop_pdms_vec import (geo_from_npz, geo_from_scene,  # noqa: E402
                                       score_batch)

OFFS = [0.0, -1.0, 1.0, -2.5, 2.5, -0.5]


def densify(anchor, n=40):
    t_src = np.linspace(0, 4.0, len(anchor))
    t_dst = np.linspace(0, 4.0, n)
    return np.stack([np.interp(t_dst, t_src, anchor[:, 0]),
                     np.interp(t_dst, t_src, anchor[:, 1])], 1).astype(np.float64)


def ego_to_world(pts, p0, h0):
    c, s = np.cos(h0), np.sin(h0)
    R = np.array([[c, -s], [s, c]], np.float64)
    return np.asarray(pts, np.float64) @ R.T + np.asarray(p0, np.float64)[None]


def make_cands(gt8, rng, sigma, g, offs=OFFS):
    base = densify(gt8)
    ramp = (np.linspace(0, 1, base.shape[0]) ** 2)[:, None]
    d = np.gradient(base, axis=0)
    nrm = np.stack([-d[:, 1], d[:, 0]], 1)
    nrm /= (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9)
    u = np.linspace(0, 1, base.shape[0])[:, None]
    S = 6 * u ** 5 - 15 * u ** 4 + 10 * u ** 3
    out = []
    for o in offs:
        path = base + o * S * nrm
        for _ in range(g):
            eps = rng.normal(0, 1.0, (2,))
            out.append(path + sigma * ramp * eps[None])
    return np.stack(out)


def spearman(a, b):
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/workspace/data/planner300/train.npz')
    ap.add_argument('--data-dir', default='/data/data_json/training')
    ap.add_argument('--n-scenes', type=int, default=300)
    ap.add_argument('--g', type=int, default=4)
    ap.add_argument('--sigma', type=float, default=0.5)
    ap.add_argument('--tol', type=float, default=1e-4)
    args = ap.parse_args()

    z = np.load(args.data, allow_pickle=True)
    N = min(args.n_scenes, len(z['ego']))
    rng = np.random.default_rng(7)
    names = np.asarray(z['scene_id'])[:N]

    d_ex, d_impl = [], []
    d_npz, sp = [], []
    gate_ex, gate_impl, gate_npz = [], [], []
    t_exact = t_vec = 0.0
    n_cand = 0
    miss_json = 0
    cover = []

    for i in range(N):
        fp = os.path.join(args.data_dir, str(names[i]))
        if not os.path.exists(fp):
            miss_json += 1
            continue
        sc = Scene(json.load(open(fp)), horizon=40)
        cand_ego = make_cands(z['gt'][i], rng, args.sigma, args.g)
        ego_w = z['ego_world'][i]
        cand_w = ego_to_world(cand_ego, ego_w[:2], ego_w[2])

        t0 = time.time()
        ex = [score_trajectory(sc, c) for c in cand_w]
        t_exact += time.time() - t0

        g_exact = geo_from_scene(sc)
        t0 = time.time()
        v_impl = score_batch(g_exact, cand_w)
        t_vec += time.time() - t0
        n_cand += len(cand_w)

        s_ex = np.array([e['driving_score'] for e in ex])
        s_im = np.asarray(v_impl['driving_score'])
        d_ex.append(np.abs(s_ex - s_im).max())
        for k in ('ep', 'ttc', 'comfort', 'lk'):
            d_impl.append(np.abs(np.array([e.get(k, 0.0) for e in ex]) -
                                 np.asarray(v_impl[k])).max())
        ge = np.array([[e['nc'], e['dac'], e['ddc']] for e in ex])
        gi = np.stack([v_impl['nc'], v_impl['dac'], v_impl['ddc']], 1)
        gate_ex.append(ge)
        gate_impl.append(gi)

        # ---- 数据集口径 ----
        if 'map_r' in z.files:
            g_npz = geo_from_npz(z, i)
            v_npz = score_batch(g_npz, cand_ego)
            s_np = np.asarray(v_npz['driving_score'])
            d_npz.append(s_np - s_ex)
            gate_npz.append(np.stack([v_npz['nc'], v_npz['dac'], v_npz['ddc']], 1))
            for k in range(len(OFFS)):
                sl = slice(k * args.g, (k + 1) * args.g)
                sp.append(spearman(s_ex[sl], s_np[sl]))
            # 走廊覆盖：候选点到最近存储线段的距离
            seg = np.asarray(z['map_r'][i, :int(z['n_map_r'][i])], np.float64)
            from metrics.openloop_pdms_vec import seg_min_dist
            dd = seg_min_dist(cand_ego.reshape(-1, 2), seg[:, :2], seg[:, 2:4])
            cover.append(float(dd.max()))

    d_ex = np.array(d_ex)
    d_impl = np.array(d_impl)
    gate_ex = np.concatenate(gate_ex)
    gate_impl = np.concatenate(gate_impl)
    print('== A. 实现对齐（同一份 Scene 几何）==')
    print('  场景 %d  候选 %d  缺失 json %d' % (N - miss_json, n_cand, miss_json))
    print('  |Δ PDMS|: max=%.3e  >%.0e 的样本数=%d'
          % (d_ex.max(), args.tol, int((d_ex > args.tol).sum())))
    print('  分量 max |Δ|: %s' % ['%.2e' % v for v in d_impl])
    print('  门控(nc/dac/ddc) 不一致: %d / %d 项'
          % (int((gate_ex != gate_impl).sum()), gate_ex.size))
    print('  耗时: exact %.1f ms/候选   vectorized %.1f ms/候选  (%.1f×)'
          % (1000 * t_exact / max(n_cand, 1), 1000 * t_vec / max(n_cand, 1),
             t_exact / max(t_vec, 1e-9)))

    if d_npz:
        d_npz = np.concatenate(d_npz)
        gate_npz = np.concatenate(gate_npz)
        sp = np.array(sp)
        print('== B. 数据集紧凑几何保真度 ==')
        print('  |Δ PDMS| vs 全量: p50=%.3e p90=%.3e max=%.3e'
              % (np.median(np.abs(d_npz)), np.percentile(np.abs(d_npz), 90),
                 np.abs(d_npz).max()))
        print('  |Δ| < %.0e 占比: %.2f%%'
              % (args.tol, 100 * np.mean(np.abs(d_npz) < args.tol)))
        print('  门控不一致: nc %d / dac %d / ddc %d  (共 %d 项)'
              % (int((gate_ex[:, 0] != gate_npz[:, 0]).sum()),
                 int((gate_ex[:, 1] != gate_npz[:, 1]).sum()),
                 int((gate_ex[:, 2] != gate_npz[:, 2]).sum()), gate_ex.size))
        print('  组内 Spearman(全量 vs 紧凑): 均值 %.3f  p10 %.3f  min %.3f'
              % (np.nanmean(sp), np.nanpercentile(sp, 10), np.nanmin(sp)))
        cover = np.array(cover)
        print('  候选点到最近存储线段的距离: p50=%.1f p90=%.1f max=%.1f m'
              % (np.median(cover), np.percentile(cover, 90), cover.max()))


if __name__ == '__main__':
    main()
