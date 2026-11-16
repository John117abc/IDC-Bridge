"""锚点集 oracle 上界：用当前地图锚点集能达到的最好 PDMS。

对每个场景，把锚点集里的每条候选都用开环 PDMS 打一遍，取最大值。
这个数字决定"规划头只要学选对"能不能成立——如果 oracle 都很低，
说明瓶颈在锚点集本身，不是选择器。

全部在 JSON 坐标系里完成（锚点从 JSON 地图生成），纯 CPU。

用法：
    python tools/oracle_anchor_pdms.py --data-dir /data/data_json/training --n-scenes 40
"""
import argparse
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from metrics.openloop_pdms import SceneCache, score_trajectory  # noqa: E402

TYPE_MAP = {'road_edge': 1, 'road_line': 2, 'lane': 3}


def _poly(arr):
    return np.asarray([[q['x'], q['y']] for q in arr], dtype=np.float64)


def json_map_rows(d):
    """把 JSON roads 折线拆成 GPUDrive 风格的线段行（x,y,len,0,0,head,type）。"""
    rows = []
    for r in (d.get('roads') or []):
        t = TYPE_MAP.get(r.get('type'))
        if t is None or len(r.get('geometry') or []) < 2:
            continue
        L = _poly(r['geometry'])
        for a, b in zip(L[:-1], L[1:]):
            mid = 0.5 * (a + b)
            ln = float(np.linalg.norm(b - a))
            if ln < 1e-6:
                continue
            h = float(np.arctan2(b[1] - a[1], b[0] - a[0]))
            rows.append([mid[0], mid[1], ln, 0.0, 0.0, h, float(t)])
    return np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 7))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--n-scenes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from env.candidate_sampler import SamplerConfig
    from env.map_candidates import generate_map_candidates

    scfg = SamplerConfig()
    cache = SceneCache(args.data_dir, n_scenes=args.n_scenes, seed=args.seed)
    print(f"场景 {len(cache.scenes)}，锚点 = {len(scfg.lat_offsets)} 横向 × "
          f"{len(scfg.anchor_lengths)} 长度 = "
          f"{len(scfg.lat_offsets)*len(scfg.anchor_lengths)} 条")

    import glob
    import json
    files = sorted(glob.glob(os.path.join(args.data_dir, 'tfrecord-*.json')))
    rng = np.random.default_rng(args.seed)
    if len(files) > args.n_scenes:
        files = [files[i] for i in rng.choice(len(files), args.n_scenes,
                                              replace=False)]
    raw = []
    for fp in files:
        try:
            with open(fp) as f:
                raw.append(json.load(f))
        except Exception:
            raw.append(None)

    oracle, mean_all, n_cand = [], [], []
    win_off, win_len, center_pdms = [], [], []
    for sc, d in zip(cache.scenes, raw):
        if d is None or len(sc.expert_pos) < 20:
            continue
        rows = json_map_rows(d)
        if len(rows) == 0:
            continue
        ex = sc.expert_pos
        ok = np.all(np.abs(ex) < 1e4, axis=1)
        if not ok.all():
            continue
        pose = (float(ex[0, 0]), float(ex[0, 1]), float(sc.expert_heading[0]))
        h0 = float(sc.expert_heading[0])
        d0 = ex[min(1, len(ex) - 1)] - ex[0]
        if np.linalg.norm(d0) > 0.05:
            pose = (pose[0], pose[1], float(np.arctan2(d0[1], d0[0])))
        # 当前速度（取日志首帧位移，因果可观测）
        v0 = float(np.linalg.norm(ex[1] - ex[0]) / 0.1) if len(ex) > 1 else 5.0
        cands, _base = generate_map_candidates(pose, rows, scfg.lat_offsets, scfg,
                                               speed0=v0)
        scores = []
        for c in cands:
            r = score_trajectory(sc, c.pos)
            scores.append(r['driving_score'] if r.get('valid') else 0.0)
        if not scores:
            continue
        scores = np.asarray(scores)
        j = int(np.argmax(scores))
        oracle.append(float(scores[j]))
        mean_all.append(float(scores.mean()))
        n_cand.append(len(scores))
        win_off.append(cands[j].param['lat_offset'])
        win_len.append(cands[j].param['len_target'])
        c_idx = [i for i, c in enumerate(cands)
                 if abs(c.param['lat_offset']) < 1e-9]
        if c_idx:
            center_pdms.append(float(max(scores[i] for i in c_idx)))

    if not oracle:
        print("无有效样本")
        return
    O = np.asarray(oracle)
    print()
    print(f"样本 {len(O)}，平均候选数 {np.mean(n_cand):.0f}")
    print(f"\n【oracle PDMS（每场景取最好的锚点）】")
    print(f"  mean={O.mean():.1f}  p10={np.percentile(O,10):.1f}  "
          f"p50={np.percentile(O,50):.1f}  p90={np.percentile(O,90):.1f}  "
          f"max={O.max():.1f}")
    M = np.asarray(mean_all)
    print(f"\n【平均 PDMS（随机挑一个锚点）】 mean={M.mean():.1f}")
    if center_pdms:
        C = np.asarray(center_pdms)
        print(f"【只允许中心候选（不许横向偏移）】 mean={C.mean():.1f}")
        print(f"  → 横向模态带来的上界增益 = {O.mean()-C.mean():+.1f} 分")
    print()
    print("【获胜锚点的横向偏移分布】")
    offs = np.asarray(win_off)
    for o in sorted(set(np.round(offs, 2))):
        print(f"  o={o:+.1f} m : {100*np.mean(np.abs(offs-o)<1e-6):5.1f}%")
    print("【获胜锚点的长度分布】")
    lens = np.asarray(win_len)
    for L in sorted(set(np.round(lens, 1))):
        print(f"  L={L:6.1f} m : {100*np.mean(np.abs(lens-L)<1e-6):5.1f}%")


if __name__ == "__main__":
    main()
