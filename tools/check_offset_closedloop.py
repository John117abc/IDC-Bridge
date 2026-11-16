"""B1-闭环：偏移参考下，加 Stanley 横向反馈能否把车拉回参考？

做法
----
1. 构造 builder 后，先把 `builder.expert_pos` 沿路径法向平移 ±0.8 m，
   再调 `generate_candidate_paths` —— path / curv_ahead / road_dist / ref_tensor
   全部自动基于偏移后的参考生成。
2. 跑 91 步闭环，每步记录状态里的 `lat`（横向误差）与 `dp`（纵向误差）。
3. 对比三种控制器：
   (a) 纯 actor（基线，无反馈）
   (b) actor + Stanley 正向增益
   (c) actor + Stanley 负向增益
   Stanley 项：δ += atan(k * e / (v + eps))，e 取 t+5 的横向误差。

判据：若 (b) 或 (c) 的 |lat| 在后半段收敛到 <0.3 m，而 (a) 停在 ~0.8 m，
说明横向反馈有效、偏移参考可跟踪。

用法（容器内）：
    python tools/check_offset_closedloop.py --data-dir /data/data_json/training \
        --model-path <pth> --num-worlds 8 --seed 0 --offset 0.8
"""
import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import numpy as np  # noqa: E402

STEER_MAX = 0.6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--config", default=os.path.join(_ROOT, "src", "configs", "eval.yaml"))
    ap.add_argument("--num-worlds", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--offset", type=float, default=0.8)
    ap.add_argument("--k", type=float, default=1.2, help="Stanley 比例增益")
    args = ap.parse_args()

    import torch

    from gpudrive.env.dataset import SceneDataLoader
    from gpudrive.env.env_torch import GPUDriveTorchEnv

    from agents.idc_agent import DiscreteIDCAgent
    from env.env_utils import (extend_action_to_3d, get_ego_indices,
                               get_env_config, load_scenes)
    from env.idc_state_builder import GPUDriveObservationBuilder
    from utils.config import build_config

    overrides = {"num_worlds": args.num_worlds, "seed": args.seed,
                 "device": "cuda", "model_path": args.model_path,
                 "load_model": True, "gif_enabled": False}
    config = build_config(args.config, cli_overrides=overrides)

    loader = SceneDataLoader(root=args.data_dir, batch_size=config.num_worlds,
                             dataset_size=config.num_worlds,
                             sample_with_replacement=False, shuffle=True,
                             seed=args.seed)
    env = GPUDriveTorchEnv(config=get_env_config(), data_loader=loader,
                           max_cont_agents=1, device=config.device,
                           action_type="continuous")
    scenes = load_scenes(env)
    builder = GPUDriveObservationBuilder(env, scenes)
    ego_indices = get_ego_indices(env, config.num_worlds)

    # ---- 偏移 expert_pos ----
    ep = builder.expert_pos
    shape = tuple(ep.shape)
    ep_np = ep.detach().cpu().numpy().copy()          # [W, A, T, 2]
    W, A, T, _ = ep_np.shape
    for w in range(W):
        for a in range(A):
            p = ep_np[w, a]
            ok = (np.abs(p[:, 0]) < 1e3) & (np.abs(p[:, 1]) < 1e3)
            if ok.sum() < 5:
                continue
            q = p[ok]
            d = np.gradient(q, axis=0)
            t = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
            n = np.stack([-t[:, 1], t[:, 0]], axis=1)
            p_off = p.copy()
            p_off[ok] = q + args.offset * n
            ep_np[w, a] = p_off
    builder.expert_pos = torch.as_tensor(ep_np, device=ep.device, dtype=ep.dtype)

    builder.generate_candidate_paths(ego_indices, num_paths=1,
                                     use_expert_path=True)
    # 断言偏移确实生效
    chk = builder.candidate_paths[0][ego_indices[0]][0]["pos"]
    print(f"偏移后参考首点=({chk[0,0]:.2f},{chk[0,1]:.2f})  "
          f"原始首点=({ep_np[0, ego_indices[0], 0, 0]:.2f},"
          f"{ep_np[0, ego_indices[0], 0, 1]:.2f})")

    agent = DiscreteIDCAgent(env, config, config.device, builder, ego_indices)
    agent.load(config.model_path)

    ref_start = agent.DIM_EGO + agent.DIM_OTHERS + agent.DIM_VALIDITY
    IX_DP, IX_LAT5, IX_DPHI5 = ref_start, ref_start + 3, ref_start + 4
    IX_V = 2

    def rollout(mode, k):
        env.reset()
        lat_hist, dp_hist = [], []
        for step in range(T):
            states = builder.get_idc_observations_batch(
                ego_indices, path_indices=[0] * config.num_worlds)
            lat = np.array([float(s[IX_LAT5]) for s in states])
            dp = np.array([float(s[IX_DP]) for s in states])
            v = np.array([float(s[IX_V]) for s in states])
            lat_hist.append(lat)
            dp_hist.append(dp)

            acts = agent.select_action(states, deterministic=True)  # [W,A,2]=[acc,steer]
            acts = acts.clone()
            if mode != "none":
                e = np.clip(lat, -5.0, 5.0)
                corr = np.arctan(k * e / np.maximum(v, 1.0))
                corr_t = torch.as_tensor(corr, device=acts.device,
                                         dtype=acts.dtype).view(-1, 1)
                acts[..., 1] = torch.clamp(acts[..., 1] + corr_t,
                                           -STEER_MAX, STEER_MAX)
            env.step_dynamics(extend_action_to_3d(acts))
        return np.asarray(lat_hist), np.asarray(dp_hist)

    print(f"\n偏移量 = {args.offset:+.2f} m，世界数 = {config.num_worlds}，"
          f"k = {args.k}")
    print(f"{'控制器':22s} {'前半|lat|':>10s} {'后半|lat|':>10s} "
          f"{'后半|dp|':>9s} {'收敛?':>6s}")
    results = {}
    for name, mode, k in (("(a) 纯 actor", "none", 0.0),
                          ("(b) +Stanley (k>0)", "add", +args.k),
                          ("(c) +Stanley (k<0)", "add", -args.k)):
        lat_h, dp_h = rollout(mode, k)
        a_lat = np.abs(lat_h)
        first = float(np.nanmean(a_lat[:20]))
        last = float(np.nanmean(a_lat[-20:]))
        last_dp = float(np.nanmean(np.abs(dp_h[-20:])))
        ok = "是" if last < 0.35 else "否"
        results[name] = last
        print(f"{name:22s} {first:10.3f} {last:10.3f} {last_dp:9.3f} {ok:>6s}")

    print()
    best = min(results, key=results.get)
    print(f"末段 |lat| 最小: {best}  ({results[best]:.3f} m)")
    if results[best] < 0.35:
        print("=> 横向反馈有效，偏移参考可以被跟踪")
    else:
        print("=> 三种控制器都收敛不到 0.35 m 以内，需要检查 Stanley 符号/增益")
        print("   或考虑训练时随机候选 warmup（方案 2）")


if __name__ == "__main__":
    main()
