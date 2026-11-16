"""B1：跟踪器对"横向误差"到底有没有响应？

背景
----
要用"候选参考路径"引入横向自由度，前提是 **GEP actor 能跟踪偏移后的参考**。
基线跟踪单条 expert 路径的误差是 0.3~0.5 m；但如果把参考横向平移 0.8 m，
策略会不会去纠偏？如果它对横向误差根本不响应，那"选哪条候选"就毫无意义
（选了也走不到），整条横向路线要改成"把横向并入动作空间"。

方法（开环灵敏度，不需要训练、不需要跑闭环）
----------------------------------------
状态里 ref_error 块含 3 个时刻的横向误差 lat（t+5/10/15）：
    ref_start = DIM_EGO + DIM_OTHERS + DIM_VALIDITY
    lat 下标 = ref_start + 3, +7, +11
把这三个位置同时加 Δ（模拟"自车相对参考偏了 Δ 米"），重算动作，看转向变化。

判据
----
* 若 |Δsteer| 随 Δ 单调增长、且对 ±Δ 反对称 → 策略在纠偏，横向自由度可用
* 若 |Δsteer| ≈ 0 → 策略不响应横向误差，横向候选路线不可行
"""
import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--config", default=os.path.join(_ROOT, "src", "configs", "eval.yaml"))
    ap.add_argument("--num-worlds", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deltas", default="0,0.2,0.4,0.6,0.8,1.2,1.6")
    args = ap.parse_args()

    from gpudrive.env.dataset import SceneDataLoader
    from gpudrive.env.env_torch import GPUDriveTorchEnv

    from agents.idc_agent import DiscreteIDCAgent
    from env.env_utils import get_ego_indices, get_env_config, load_scenes
    from env.idc_state_builder import GPUDriveObservationBuilder
    from utils.config import build_config

    overrides = {
        "num_worlds": args.num_worlds,
        "seed": args.seed,
        "device": "cuda",
        "model_path": args.model_path,
        "load_model": True,
        "gif_enabled": False,
    }
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
    builder.generate_candidate_paths(ego_indices, num_paths=1,
                                     use_expert_path=True)
    agent = DiscreteIDCAgent(env, config, config.device, builder, ego_indices)
    agent.load(config.model_path)

    env.reset()
    states = builder.get_idc_observations_batch(
        ego_indices, path_indices=[0] * config.num_worlds)

    ref_start = agent.DIM_EGO + agent.DIM_OTHERS + agent.DIM_VALIDITY
    lat_idx = [ref_start + 3, ref_start + 7, ref_start + 11]
    print(f"状态维度 {states[0].shape[0]}  ref_start={ref_start}  "
          f"lat 下标={lat_idx}")

    deltas = [float(x) for x in args.deltas.split(",")]

    # 注意：不能用 agent.select_action —— 它每次调用会推进内部滚动窗口，
    # 重复喂同一个 state 也会得到不同动作。这里直接走确定性的 _encode + _policy，
    # 并且必须把 actor 切到 eval（否则 dropout 会引入 ~0.02 rad 的噪声，
    # 实测 Δ=0 时 |Δsteer| 就有 0.02，完全盖过信号）。
    import torch

    agent.actor.eval()
    dev = next(agent.actor.parameters()).device

    def act_of(st_list):
        s_np = np.stack([np.asarray(s, dtype=np.float32) for s in st_list])
        with torch.no_grad():
            s_t = torch.as_tensor(s_np, device=dev)
            # _policy 内部会调 _encode，所以这里喂 66 维原始状态
            win = s_t[:, None, :].expand(-1, 30, -1).contiguous()
            raw = agent._policy(win)                     # [W, 2] = [steer_raw, acc_raw]
            _a, delta = agent._physical_action(raw)      # 物理前轮角（rad）
        return delta.detach().cpu().numpy()

    base = act_of(states)                       # [W] 物理前轮角
    print(f"基线物理 steer: mean={base.mean():+.4f} rad  std={base.std():.4f}  "
          f"range=[{base.min():+.3f}, {base.max():+.3f}]")
    print()

    print("Δlat(m)   mean Δsteer(rad)   mean|Δsteer|   ")
    rows = []
    for d in sorted(deltas):
        st2 = [s.copy() for s in states]
        for w in range(config.num_worlds):
            for i in lat_idx:
                st2[w][i] = st2[w][i] + d
        a2 = act_of(st2)
        dsteer = a2 - base
        rows.append((d, float(dsteer.mean()), float(np.abs(dsteer).mean())))
        print(f"{d:+.2f}     {dsteer.mean():+.5f}          {np.abs(dsteer).mean():.5f}")

    # 过原点的最小二乘斜率 + R²
    ds_ = np.asarray([r[0] for r in rows])
    ms_ = np.asarray([r[1] for r in rows])
    slope = float((ds_ * ms_).sum() / max((ds_ ** 2).sum(), 1e-12))
    ss_res = float(((ms_ - slope * ds_) ** 2).sum())
    ss_tot = float((ms_ ** 2).sum())
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    print()
    print(f"过原点拟合: d(steer)/d(lat) = {slope:+.4f} rad/m "
          f"({np.rad2deg(slope):+.2f} deg/m)   R²={r2:.3f}")

    # 判读
    # 参考量级：要把 0.8 m 的横向偏移在 30 m 纵向内收敛，需要的曲率
    #   κ ≈ 2e/L² = 2*0.8/900 = 0.0018 1/m
    #   δ ≈ atan(κ*L_wb) = atan(0.0018*5.286) = 0.0094 rad
    need = float(np.arctan(2 * 0.8 / 900.0 * 5.286))
    print()
    print(f"参考：收敛 0.8 m 偏移（30 m 内）约需 δ ≈ {need:.4f} rad "
          f"({np.rad2deg(need):.2f} deg)")
    print(f"      当前灵敏度 × 0.8 m = {abs(slope)*0.8:.4f} rad  "
          f"→ 需求量的 {abs(slope)*0.8/max(need,1e-9):.1f} 倍")
    print()
    if r2 < 0.3:
        print("=> 响应与横向误差几乎无线性关系（R²<0.3），可能是窗口被固定成")
        print("   常量导致策略落在分布外；需要在真实历史窗口上重测")
    elif abs(slope) * 0.8 < need:
        print("=> 灵敏度不足以跟上 0.8 m 偏移，横向候选会跟不动")
    else:
        print("=> 灵敏度足够纠偏，横向候选参考路线可行")


if __name__ == "__main__":
    main()
