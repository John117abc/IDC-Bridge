"""导师要求的关键对照实验（一次遍历，记录每个候选的 plan/exec 双判据）。

三组：
  G0 = top-1
  G1 = 规划轨迹筛选（路缘 + OBB，周车用**因果预测**）
  G2 = 推演轨迹筛选（同一套检查，但算在纯追踪推演上）

评测（独立、用真实未来）：
  碰撞：OBB 重叠（真实周车）
  越界：车中心到路缘距离 < 半车宽 + 0.1 m（整车越界代理）
  进度：沿专家未来的投影

输出：逐场景逐候选 npz + 控制台三张表
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "tools"))

from metrics.openloop_pdms import (Scene, _min_dist, _obb_overlap,  # noqa: E402
                                   step_metrics)
from build_head_dataset import build_one  # noqa: E402
from eval_head_pdms import densify, ego_to_world  # noqa: E402
from exp_pre import pure_pursuit  # noqa: E402

D_EDGE = 1.0
HALF_W = 0.975


def predict_agents(sc, k_hist=3):
    """因果预测：用 [t-k, t] 的历史速度做恒定速度外推（40 步）。"""
    t = sc.t_now
    H = sc.n_future
    k = min(k_hist, t)
    P = np.zeros((len(sc.ap), H, 2), np.float32)
    V = np.zeros((len(sc.ap),), np.float32)
    valid = np.ones((len(sc.ap), H), bool)
    for a in range(len(sc.ap)):
        p0 = sc.ap[a, t]
        v = (sc.ap[a, t] - sc.ap[a, t - k]) / (k * 0.1)
        V[a] = np.linalg.norm(v)
        for j in range(H):
            P[a, j] = p0 + v * (j + 1) * 0.1
        valid[a, :] = sc.av[a, t]
    return P, V, valid


def check_traj(sc, traj, agents_P, agents_valid, d_edge, use_foot=False):
    """返回 (ok, edge_clear, coll)。agents_P 可以是预测或真实未来。"""
    H = min(len(traj), sc.n_future)
    traj = traj[:H]
    if len(sc.ea):
        edge_clear = min(_min_dist(p, sc.ea, sc.eb) for p in traj)
    else:
        edge_clear = 1e9
    ok_edge = edge_clear >= d_edge
    coll = False
    for tt in range(H):
        p = traj[tt]
        if tt + 1 < H:
            dv = traj[tt + 1] - p
        else:
            dv = p - traj[tt - 1]
        h = float(np.arctan2(dv[1], dv[0])) if np.linalg.norm(dv) > 1e-6 else 0.0
        for a in range(len(agents_P)):
            if not agents_valid[a, tt]:
                continue
            q = agents_P[a, tt]
            if np.linalg.norm(q - p) > 6.0:
                continue
            if _obb_overlap(p, h, sc.ego_dim, q, h, sc.ad[a]):
                coll = True
                break
        if coll:
            break
    return (ok_edge and not coll), float(edge_clear), coll


def exec_eval(sc, traj):
    """评测：撞真实周车 / 整车越界 / 进度。"""
    H = min(len(traj), sc.n_future)
    traj = traj[:H]
    m = step_metrics(sc, traj)
    coll = bool(np.any(m['coll']))
    off = False
    if len(sc.ea):
        for p in traj:
            if _min_dist(p, sc.ea, sc.eb) < HALF_W + 0.1:
                off = True
                break
    prog = float(sc.progress(traj[-1]))
    return coll, off, prog


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--files-json', required=True)
    ap.add_argument('--n-scenes', type=int, default=2000)
    ap.add_argument('--k-top', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='/workspace/data/screen')
    args = ap.parse_args()

    import torch
    from train_head import Head
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

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

    names = json.load(open(args.files_json))
    rng = np.random.default_rng(args.seed)
    if len(names) > args.n_scenes:
        names = [names[i] for i in rng.choice(len(names), args.n_scenes,
                                              replace=False)]
    print(f'评测场景 {len(names)}（来自 {args.files_json}）', flush=True)

    rows = []          # 每条候选一行
    scene_ids = []
    for i, n in enumerate(names):
        fp = os.path.join(args.data_dir, n)
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        sc = Scene(d, horizon=40)
        s = build_one(d, A)
        if s is None or sc.n_future < 20:
            continue
        p0, h0 = sc.expert_pos[sc.t_now], float(sc.expert_heading[sc.t_now])
        with torch.no_grad():
            ego = torch.tensor((s['ego'] - ego_m) / ego_s, device=dev)[None]
            ag = torch.tensor((s['agent'] - ag_m) / ag_s, device=dev)[None]
            am_ = torch.tensor(s['amask'], device=dev)[None]
            mp_ = torch.tensor((s['map'] - mp_m) / mp_s, device=dev)[None]
            mm_ = torch.tensor(s['mmask'], device=dev)[None]
            logits, _ = model(ego, ag, am_, mp_, mm_, anchors)
            prob = torch.softmax(logits, -1)[0].cpu().numpy()
        # 因果预测周车（只用于筛选）
        pred_P, pred_V, pred_valid = predict_agents(sc)
        v0 = max(sc.speed_now(), 1.0)
        scene_ids.append(n)
        for r, k in enumerate(np.argsort(-prob)[:args.k_top]):
            plan = ego_to_world(densify(A[k], sc.n_future), p0, h0)
            try:
                roll = pure_pursuit(plan, v0)
            except Exception:
                roll = plan
            H = sc.n_future
            if len(roll) < H:
                roll = np.vstack([roll, np.repeat(roll[-1:], H - len(roll), 0)])
            roll = roll[:H]
            # 筛选判据（预测周车）
            ok_plan, ec_plan, _ = check_traj(sc, plan, pred_P, pred_valid, D_EDGE)
            ok_exec, ec_exec, _ = check_traj(sc, roll, pred_P, pred_valid, D_EDGE)
            # 评测（真实未来）
            coll, off, prog = exec_eval(sc, roll)
            rows.append(dict(scene=len(scene_ids) - 1, rank=r, k=int(k),
                             prob=float(prob[k]), ok_plan=bool(ok_plan),
                             ok_exec=bool(ok_exec), coll=bool(coll),
                             off=bool(off), prog=float(prog),
                             ec_plan=float(ec_plan), ec_exec=float(ec_exec)))
        if (i + 1) % 200 == 0:
            print(f'  {i+1}/{len(names)}', flush=True)

    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out, 'screen.npz'),
                        rows=json.dumps(rows), scene_ids=np.array(scene_ids))
    print('saved', os.path.join(args.out, 'screen.npz'), flush=True)

    # ---------- 分析 ----------
    by_scene = {}
    for r in rows:
        by_scene.setdefault(r['scene'], []).append(r)

    def pick(scene_rows, key):
        for r in sorted(scene_rows, key=lambda x: x['rank']):
            if r[key]:
                return r
        return sorted(scene_rows, key=lambda x: x['rank'])[0]

    res = {g: dict(coll=0, off=0, prog=[], n=0, allfail=0, changed=0)
           for g in ('G0', 'G1', 'G2')}
    for sid, srows in by_scene.items():
        g0 = pick(srows, 'rank' == 0 or True)     # top-1
        g0 = sorted(srows, key=lambda x: x['rank'])[0]
        g1 = pick(srows, 'ok_plan')
        g2 = pick(srows, 'ok_exec')
        for g, r in (('G0', g0), ('G1', g1), ('G2', g2)):
            res[g]['n'] += 1
            res[g]['coll'] += int(r['coll'])
            res[g]['off'] += int(r['off'])
            res[g]['prog'].append(r['prog'])
        if not any(x['ok_plan'] for x in srows):
            res['G1']['allfail'] += 1
        if not any(x['ok_exec'] for x in srows):
            res['G2']['allfail'] += 1
        if g2['rank'] != g0['rank']:
            res['G2']['changed'] += 1

    n = len(by_scene)
    print(f'\n=== 场景级（n={n}）===')
    print(f"{'组':4s} {'碰撞':>6s} {'越界':>6s} {'平均进度':>8s} {'全拒':>6s} {'改选':>6s}")
    for g in ('G0', 'G1', 'G2'):
        r = res[g]
        print(f'{g:4s} {r["coll"]:6d} {r["off"]:6d} {np.mean(r["prog"]):8.3f} '
              f'{r["allfail"]:6d} {r["changed"]:6d}')

    # 候选级四分类（plan 判据 vs exec 结局）
    pp = sum(1 for r in rows if r['ok_plan'] and not (r['coll'] or r['off']))
    pe = sum(1 for r in rows if r['ok_plan'] and (r['coll'] or r['off']))
    fp = sum(1 for r in rows if not r['ok_plan'] and not (r['coll'] or r['off']))
    fe = sum(1 for r in rows if not r['ok_plan'] and (r['coll'] or r['off']))
    print(f'\n=== 候选级四分类（共 {len(rows)} 条）===')
    print(f'  PPEP (plan过/exec过): {pp}')
    print(f'  PPEF (plan过/exec失败): {pe}   ← 核心')
    print(f'  PFEP (plan拒/exec过): {fp}   ← 误杀')
    print(f'  PFEF (plan拒/exec失败): {fe}')

    # PPEF 场景的挽回
    ppef_scenes = [sid for sid, sr in by_scene.items()
                   if any(r['ok_plan'] and (r['coll'] or r['off']) for r in sr)
                   and sorted(sr, key=lambda x: x['rank'])[0]['ok_plan']]
    if ppef_scenes:
        c1 = sum(1 for sid in ppef_scenes
                 if pick(by_scene[sid], 'ok_plan')['coll'] or
                 pick(by_scene[sid], 'ok_plan')['off'])
        c2 = sum(1 for sid in ppef_scenes
                 if pick(by_scene[sid], 'ok_exec')['coll'] or
                 pick(by_scene[sid], 'ok_exec')['off'])
        print(f'\n=== PPEF 场景（G1 选中且失败）: {len(ppef_scenes)} ===')
        print(f'  G1 失败 {c1}，G2 失败 {c2}，挽回 {c1-c2} '
              f'({100*(c1-c2)/max(c1,1):.0f}%)')

    # 图
    fig, ax = plt.subplots(figsize=(7, 4), dpi=160)
    xs = ['G0\ntop-1', 'G1\nplan screen', 'G2\nrollout screen']
    coll = [res[g]['coll'] for g in ('G0', 'G1', 'G2')]
    off = [res[g]['off'] for g in ('G0', 'G1', 'G2')]
    x = np.arange(3)
    ax.bar(x - 0.18, coll, 0.34, label='collision', color='#e53935')
    ax.bar(x + 0.18, off, 0.34, label='off-road', color='#fb8c00')
    for i in range(3):
        ax.text(x[i] - 0.18, coll[i] + 0.3, str(coll[i]), ha='center', fontsize=9)
        ax.text(x[i] + 0.18, off[i] + 0.3, str(off[i]), ha='center', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(xs, fontsize=9)
    ax.set_ylabel(f'scenes (n={n})')
    ax.set_title('Candidate screening: plan-level vs rollout-level')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(f'{args.out}/fig_screen.png')
    plt.close(fig)
    print('图已保存', args.out)


if __name__ == '__main__':
    main()
