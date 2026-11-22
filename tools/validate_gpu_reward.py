"""验证 GPU 版 PDMS 与 numpy 版逐条一致（<1e-4）。"""
import sys
import time

import numpy as np
import torch

sys.path.insert(0, '/gpudrive/baselines/IDC-Bridge/src')
sys.path.insert(0, '/gpudrive/baselines/IDC-Bridge/tools')

from metrics.openloop_pdms_vec import geo_from_npz, score_batch  # noqa: E402
from metrics.openloop_pdms_torch import geo_from_npz_t, score_batch_t  # noqa: E402
from validate_pdms_vec import make_cands  # noqa: E402


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    z = np.load('/workspace/data/planner60k/val.npz', allow_pickle=True)
    n = min(n, len(z['gt']))
    rng = np.random.default_rng(0)
    dmax, parts = 0.0, []
    t_np = t_t = 0.0
    for i in range(n):
        cands = make_cands(z['gt'][i], rng, 0.5, 4)
        g_np = geo_from_npz(z, i)
        t0 = time.time()
        a = score_batch(g_np, cands)
        t_np += time.time() - t0
        g_t = geo_from_npz_t(z, i, dev)
        ct = torch.as_tensor(cands, dtype=torch.float64, device=dev)
        t0 = time.time()
        b = score_batch_t(g_t, ct)
        t_t += time.time() - t0
        da = np.abs(np.asarray(a['driving_score']) - b['driving_score'].cpu().numpy())
        dmax = max(dmax, da.max())
        parts.append([np.abs(np.asarray(a[k]) - b[k].cpu().numpy()).max()
                      for k in ('ep', 'ttc', 'comfort', 'lk')])
        ga = np.stack([a['nc'], a['dac'], a['ddc']], 1)
        gb = torch.stack([b['nc'], b['dac'], b['ddc']], 1).cpu().numpy()
        if not np.array_equal(ga, gb):
            print('门控不一致 scene %d: %s vs %s' % (i, ga.tolist(), gb.tolist()))
    parts = np.array(parts)
    print('== GPU 版 vs numpy 版（%d 场景 × 24 候选，device=%s）==' % (n, dev))
    print('  max |ΔPDMS| = %.3e' % dmax)
    print('  分量 max |Δ|: ep %.2e ttc %.2e comfort %.2e lk %.2e'
          % tuple(parts.max(0)))
    print('  耗时: numpy %.1f ms/场景 | GPU %.1f ms/场景'
          % (1000 * t_np / n, 1000 * t_t / n))


if __name__ == '__main__':
    main()
