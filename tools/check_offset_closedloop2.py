"""B1-闭环 v2：偏移参考（同起点、不同终点）下，Stanley 能否压住跟踪误差？

与 v1 的区别
------------
v1 用"平行偏移"（起点就偏 o），测出来的是阶跃响应，而那不是候选要求的语义。
v2 用 candidate_sampler 的 **smoothstep 横向剖面**：起点 = ego 位姿，终点偏 o。

注入方式
--------
把候选轨迹写回 `builder.expert_pos`，再调 `generate_candidate_paths` ——
builder 会基于它重建 path / curv_ahead / road_dist / ref_tensor，
不需要复制 builder 内部逻辑。

注意：builder 的路径构建会 `insert(0, ego_pose)`，即参考首点始终是 ego 当前位置，
这和我们的候选"同起点"语义一致。

用法（容器内）：
    python tools/check_offset_closedloop2.py --data-dir /data/data_json/training \
        --model-path <pth> --num-worlds 8 --seed 0 --offset 1.6
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
    ap.add_argument("--offset", type=float, default=1.6)
    ap.add_argument("--k", type=float, default=1.2)
    args = ap.parse_args()

    import torch

    from gpudrive.env.dataset import SceneDataLoader
    from gpudrive.env.env_torch import GPUDriveTorchEnv

    from agents.idc_agent import DiscreteIDCAgent
    from env.candidate_sampler import SamplerConfig, boundary_segments, make_candidate
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

    maps = env.sim.map_observation_tensor().to_torch().cpu().numpy()
    ep = builder.expert_pos
    ep_np = ep.detach().cpu().numpy().copy()          # [W, A, T, 2]
    W, A, T, _ = ep_np.shape
    cfg = SamplerConfig()

    # ---- 用采样器生成候选，并写回 expert_pos ----
    n_ok = 0
    for w in range(W):
        a = ego_indices[w]
        p = ep_np[w, a]
        ok = (np.abs(p[:, 0]) < 1e3) & (np.abs(p[:, 1]) < 1e3)
        if not ok.all():
            # 只保留完全有效的轨迹，否则候选剖面会跨空洞
            continue
        a_seg, b_seg = boundary_segments(maps[w])
        cand = make_candidate(p, args.offset, 0.0, a_seg, b_seg, cfg)
        # 校验：起点必须与 ego 一致
        err0 = float(np.linalg.norm(cand.pos[0] - p[0]))
        ep_np[w, a] = cand.pos
        n_ok += 1
        if w == 0:
            print(f"候选起点误差={err0:.6f} m  终点横向偏移="
                  f"{cand.pos[-1,1]-p[-1,1]:+.2f} m")
    builder.expert_pos = torch.as_tensor(ep_np, device=ep.device, dtype=ep.dtype)
    print(f"注入候选的世界数: {n_ok}/{W}，偏移 = {args.offset:+.2f} m")

    builder.generate_candidate_paths(ego_indices, num_paths=1,
                                     use_expert_path=True)

    agent = DiscreteIDCAgent(env, config, config.device, builder, ego_indices)
    agent.load(config.model_path)

    ref_start = agent.DIM_EGO + agent.DIM_OTHERS + agent.DIM_VALIDITY
    IX_DP, IX_LAT5, IX_V = ref_start, ref_start + 3, 2

    def rollout(mode, k):
        env.reset()
        lat_h, dp_h = [], []
        for step in range(T):
            states = builder.get_idc_observations_batch(
                ego_indices, path_indices=[0] * config.num_worlds)
            lat = np.array([float(s[IX_LAT5]) for s in states])
            dp = np.array([float(s[IX_DP]) for s in states])
            v = np.array([float(s[IX_V]) for s in states])
            lat_h.append(lat)
            dp_h.append(dp)
            acts = agent.select_action(states, deterministic=True).clone()
            if mode != "none":
                e = np.clip(lat, -5.0, 5.0)
                corr = np.arctan(k * e / np.maximum(v, 1.0))
                ct = torch.as_tensor(corr, device=acts.device,
                                     dtype=acts.dtype).view(-1, 1)
                acts[..., 1] = torch.clamp(acts[..., 1] + ct, -STEER_MAX, STEER_MAX)
            env.step_dynamics(extend_action_to_3d(acts))
        return np.asarray(lat_h), np.asarray(dp_h)

    def valid_stats(h):
        """过滤哨兵/跑飞的世界：|dp|>200 或 |lat|>100 视为无效。"""
        m = (np.abs(h) < 200.0)
        return m

    print()
    print(f"{'控制器':22s} {'有效世界':>8s} {'前1s |lat|':>11s} "
          f"{'末1s |lat|':>11s} {'末1s |dp|':>10s}")
    out = {}
    for name, mode, k in (("(a) 纯 actor", "none", 0.0),
                          ("(b) +Stanley k>0", "add", +args.k),
                          ("(c) +Stanley k<0", "add", -args.k)):
        lat_h, dp_h = rollout(mode, k)
        m = valid_stats(dp_h) & valid_stats(lat_h)
        a_lat = np.where(m, np.abs(lat_h), np.nan)
        a_dp = np.where(m, np.abs(dp_h), np.nan)
        first = float(np.nanmean(a_lat[:10]))
        last = float(np.nanmean(a_lat[-10:]))
        lastdp = float(np.nanmean(a_dp[-10:]))
        nvalid = int(np.sum(np.all(m, axis=0)))
        out[name] = last
        print(f"{name:22s} {nvalid:8d} {first:11.3f} {last:11.3f} {lastdp:10.3f}")

    print()
    best = min(out, key=lambda x: (np.isnan(out[x]), out[x]))
    print(f"末段 |lat| 最小: {best}  ({out[best]:.3f} m)")
    if out[best] < 0.5:
        print("=> 偏移参考可以被跟踪（同起点语义下），横向候选方案成立")
    else:
        print("=> 仍跟踪不上，需要看 Stanley 符号/增益 或考虑训练时随机候选 warmup")


if __name__ == "__main__":
    main()
