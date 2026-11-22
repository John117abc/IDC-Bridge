"""排查：模态差异小，是不是因为场景本来就是直行/无歧义。

按场景"内在歧义度"分层：
  - GT 朝向变化 |Δh|（8 个 waypoint 的首段与末段夹角）
  - 路口线索：自车路径 15 m 内、与自车朝向夹角 >30° 的车道线段数（横向车道 = 交叉口/可转弯）
  - 自车速度
然后看 A / B 的模态差异在各层里是否不同。
"""
import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, '/gpudrive/baselines/IDC-Bridge/src')
sys.path.insert(0, '/gpudrive/baselines/IDC-Bridge/tools')
from train_planner import Planner  # noqa: E402


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def predict(ckpt, z, idx, dev):
    ck = torch.load(ckpt, map_location='cpu')
    m = Planner(K=ck.get('K', 6)).to(dev).eval()
    m.load_state_dict(ck['model'])
    out = {i: None for i in idx}
    with torch.no_grad():
        for b0 in range(0, len(idx), 64):
            sl = idx[b0:b0 + 64]
            p = m(torch.tensor(z['ego'][sl], dtype=torch.float32),
                  torch.tensor(z['agent'][sl], dtype=torch.float32),
                  torch.tensor(z['amask'][sl], dtype=torch.float32),
                  torch.tensor(z['map'][sl], dtype=torch.float32),
                  torch.tensor(z['mmask'][sl], dtype=torch.float32))
            for t, i in enumerate(sl):
                out[i] = p[t].numpy()
    return np.stack([out[i] for i in idx])


def scene_stats(z, idx):
    gt = z['gt'][idx]                                   # (n,8,2)
    d = np.diff(gt, axis=1)
    h = np.arctan2(d[..., 1], d[..., 0])
    dh = np.abs(wrap(h[:, -1] - h[:, 0])) * 180 / np.pi  # 末端与首段夹角（度）
    # GT 相对直线的最大横向偏离
    u = d[:, 0] / (np.linalg.norm(d[:, 0], axis=1, keepdims=True) + 1e-9)
    norm = np.stack([-u[:, 1], u[:, 0]], 1)
    lat = np.abs((gt * norm[:, None]).sum(-1)).max(1)
    # 路口线索：map_r（自车系）里与自车朝向夹角 >30° 的车道线段数（15 m 内）
    nbranch = np.zeros(len(idx))
    for t, i in enumerate(idx):
        m = int(z['n_map_r'][i])
        seg = z['map_r'][i, :m]
        lane = seg[seg[:, 4] > 0.5]
        if len(lane) == 0:
            continue
        a, b = lane[:, :2], lane[:, 2:4]
        mid = 0.5 * (a + b)
        far = np.linalg.norm(mid, axis=1) > 15
        ang = np.abs(wrap(np.arctan2(b[:, 1] - a[:, 1], b[:, 0] - a[:, 0])))
        nbranch[t] = int(np.sum((ang > np.pi / 6) & ~far))
    return dh, lat, nbranch, z['ego'][idx, 0]


def div_stats(p, mask):
    iu = np.triu_indices(p.shape[1], 1)
    stop = np.linalg.norm(p[:, iu[0], -1] - p[:, iu[1], -1], axis=-1)
    gt = None
    return dict(pair=np.median(stop), minpair=np.median(stop.min(1)),
                allgt2=100 * np.mean(stop.min(1) > 2),
                anylt1=100 * np.mean(stop.min(1) < 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/workspace/data/planner60k')
    ap.add_argument('--a', default='/workspace/data/ab60k_A/latest.pt')
    ap.add_argument('--b', default='/workspace/data/ab60k_B/latest.pt')
    ap.add_argument('--ea', default='/workspace/data/ab60k_A/eval.npz')
    ap.add_argument('--eb', default='/workspace/data/ab60k_B/eval.npz')
    ap.add_argument('--limit', type=int, default=200)
    args = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    z = np.load(f'{args.data}/val.npz', allow_pickle=True)
    idx = list(range(min(len(z['ego']), args.limit)))
    pa, pb = predict(args.a, z, idx, dev), predict(args.b, z, idx, dev)
    dh, lat, nbranch, v0 = scene_stats(z, idx)
    zz = np.array(idx)

    print('== 场景歧义度分布（%d 场景）==' % len(idx))
    print('  GT 朝向变化 |Δh|: p10 %.1f  p50 %.1f  p90 %.1f 度'
          % tuple(np.percentile(dh, [10, 50, 90])))
    print('  |Δh|<5°（近似直行）: %d 场景 (%.0f%%)；5-20°: %d；>20°: %d'
          % (int((dh < 5).sum()), 100 * np.mean(dh < 5),
             int(((dh >= 5) & (dh <= 20)).sum()), int((dh > 20).sum())))
    print('  含横向车道（路口线索）的场景: %d (%.0f%%)'
          % (int((nbranch > 3).sum()), 100 * np.mean(nbranch > 3)))
    print('  近似直行且无路口线索: %d (%.0f%%)'
          % (int(((dh < 5) & (nbranch <= 3)).sum()),
             100 * np.mean((dh < 5) & (nbranch <= 3))))

    E = np.load(args.ea)
    F = np.load(args.eb)
    iu = np.triu_indices(pa.shape[1], 1)

    def bucket(name, mask):
        if mask.sum() < 3:
            print(f'--- {name}: 场景太少 ({int(mask.sum())}) ---')
            return
        row = [name, int(mask.sum())]
        for p in (pa, pb):
            stop = np.linalg.norm(p[mask][:, iu[0], -1] - p[mask][:, iu[1], -1],
                                  axis=-1)
            row += [np.median(stop), np.median(stop.min(1)),
                    100 * np.mean(stop.min(1) > 2)]
        # 互不相同且 PDMS>0（用 eval npz）
        dm = []
        for ev in (E, F):
            pm = ev['pdms'][zz][mask]
            stop = np.linalg.norm(pa[mask][:, iu[0], -1] - pa[mask][:, iu[1], -1],
                                  axis=-1) * 0  # placeholder
            dm.append(pm)
        print('  %-22s n=%-3d | A 两两 %5.1f 最小 %5.2f 全>2m %3.0f%% | '
              'B 两两 %5.1f 最小 %5.2f 全>2m %3.0f%%'
              % tuple(row))

    print('\n== 按 GT 朝向变化分层：模态两两终点距离 ==')
    bucket('|Δh| < 5° (直行)', dh < 5)
    bucket('5° ≤ |Δh| ≤ 20°', (dh >= 5) & (dh <= 20))
    bucket('|Δh| > 20° (转弯)', dh > 20)
    print('\n== 按路口线索分层 ==')
    bucket('nbranch ≤ 3 (无路口)', nbranch <= 3)
    bucket('nbranch > 3 (有路口)', nbranch > 3)
    print('\n== 交叉：直行且无路口 vs 其余 ==')
    bucket('直行且无路口', (dh < 5) & (nbranch <= 3))
    bucket('其余', ~((dh < 5) & (nbranch <= 3)))

    # 相关性：歧义度 vs 模态分散度
    for tag, p in (('A', pa), ('B', pb)):
        stop = np.linalg.norm(p[:, iu[0], -1] - p[:, iu[1], -1], axis=-1)
        mine = stop.min(1)
        print('\n%s: 最小对距离 与 |Δh| 的相关 %.3f，与 nbranch 的相关 %.3f'
              % (tag, np.corrcoef(mine, dh)[0, 1], np.corrcoef(mine, nbranch)[0, 1]))
        print('   |Δh|<5° 场景里最小对 p50 %.2f m；|Δh|>20° 场景里 %.2f m'
              % (np.median(mine[dh < 5]), np.median(mine[dh > 20])))


if __name__ == '__main__':
    main()
