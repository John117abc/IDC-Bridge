"""隔离诊断：PDMS 掉到 ~45 是"几何"还是"速度"造成的？

四组对照（几何 × 速度）：
  A  地图几何 + 地图速度（现状）
  B  地图几何 + 专家速度
  C  专家几何 + 地图速度
  D  专家几何 + 专家速度（≈ 之前的 70 分基线）

只报客观量，不依赖 PDMS 打分器：碰撞世界数/步数、压线世界数/步数、
完成度（末段投影进度）、跟踪误差。

用法（容器内）：
    python tools/diag_speed_geometry.py --data-dir /data/data_json/training \
        --model-path <pth> --num-worlds 32 --seed 0
"""
import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import numpy as np  # noqa: E402

OFFSETS_EXPERT = (-0.8, 0.0, 0.8)
CONST_SPEED = 10.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--config", default=os.path.join(_ROOT, "src", "configs", "eval.yaml"))
    ap.add_argument("--num-worlds", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--geo", choices=["map", "expert"], default="map")
    ap.add_argument("--speed", choices=["ref", "expert", "const"], default="ref")
    args = ap.parse_args()

    from gpudrive.env.dataset import SceneDataLoader
    from gpudrive.env.env_torch import GPUDriveTorchEnv

    from agents.idc_agent import DiscreteIDCAgent
    from env.candidate_sampler import SamplerConfig, curvature, install_fixed
    from env.env_utils import (extend_action_to_3d, get_ego_indices,
                               get_env_config, load_scenes)
    from env.idc_state_builder import GPUDriveObservationBuilder
    from env.map_candidates import install_map_candidates
    from utils.config import build_config

    cfg = build_config(args.config, cli_overrides={
        "num_worlds": args.num_worlds, "seed": args.seed, "device": "cuda",
        "model_path": args.model_path, "load_model": True, "gif_enabled": False})
    loader = SceneDataLoader(root=args.data_dir, batch_size=cfg.num_worlds,
                             dataset_size=cfg.num_worlds,
                             sample_with_replacement=False, shuffle=True,
                             seed=args.seed)
    env = GPUDriveTorchEnv(config=get_env_config(), data_loader=loader,
                           max_cont_agents=1, device=cfg.device,
                           action_type="continuous")
    scenes = load_scenes(env)
    builder = GPUDriveObservationBuilder(env, scenes)
    ego = get_ego_indices(env, cfg.num_worlds)
    W = cfg.num_worlds
    maps = env.sim.map_observation_tensor().to_torch().cpu().numpy()
    expert = builder.expert_pos.detach().cpu().numpy()
    builder.generate_candidate_paths(ego, num_paths=1, use_expert_path=True)
    scfg = SamplerConfig()

    abs_np = env.sim.absolute_self_observation_tensor().to_torch().cpu().numpy()
    rel_np = env.sim.self_observation_tensor().to_torch().cpu().numpy()
    poses = [(float(abs_np[w, ego[w], 0]), float(abs_np[w, ego[w], 1]),
              float(abs_np[w, ego[w], 7])) for w in range(W)]
    speeds = [float(np.clip(rel_np[w, ego[w], 2], 1.0, 20.0)) for w in range(W)]

    agent = DiscreteIDCAgent(env, cfg, cfg.device, builder, ego)
    agent.load(cfg.model_path)
    ref_start = agent.DIM_EGO + agent.DIM_OTHERS + agent.DIM_VALIDITY
    IX_DP, IX_LAT5 = ref_start, ref_start + 3
    IX_IDX = ref_start + agent.DIM_REF_ERROR
    T = builder.EXPERT_TRAJ_LEN

    def override_speed(mode):
        """把每条候选的 speed 覆盖成 const / expert 速度剖面。"""
        for w in range(W):
            a = int(ego[w])
            if mode == 'expert':
                ex = expert[w, a]
                ok = np.all(np.abs(ex) < 1e4, axis=1)
                ex = ex[ok]
                if len(ex) < 3:
                    continue
                sp = np.r_[np.linalg.norm(np.diff(ex, axis=0), axis=1) / 0.1, 0.0]
                # 重采样到 91 点
                idx = np.linspace(0, len(sp) - 1, 91)
                v = np.interp(idx, np.arange(len(sp)), sp)
            else:                                   # const
                v = None
            for k, p in enumerate(builder.candidate_paths[w][a]):
                if mode == 'const':
                    kap = curvature(np.asarray(p['pos'], dtype=np.float64), 0.1)
                    vk = np.minimum(CONST_SPEED, np.sqrt(scfg.a_lat_max
                                                        / np.maximum(kap, 1e-6)))
                else:
                    vk = v
                p['speed'] = np.clip(vk, 0.0, scfg.v_cap).astype(np.float32)
        builder.rebuild_paths_tensor()

    def run_once(geo, spd):
        # ---- 安装候选 ----
        if geo == 'map':
            K, _pw = install_map_candidates(builder, ego, maps, poses, speeds,
                                            scfg, k_select=3)
        else:
            K = install_fixed(builder, ego, maps, OFFSETS_EXPERT, scfg)

        # 用第 1 条（expert 顺序 −0.8,0,0.8 的中间；map 顺序按 |o| 排序的中心）
        cidx = 1 if geo == 'expert' else 0
        if spd != 'ref':
            override_speed(spd)

        if hasattr(agent, 'state_windows'):
            agent.state_windows.clear()
        for fn in ('reset', 'begin_epoch'):
            if hasattr(builder, fn):
                try:
                    getattr(builder, fn)()
                except Exception:
                    pass

        env.reset()
        ncol = np.zeros(W, dtype=bool)
        noff = np.zeros(W, dtype=bool)
        col_steps = off_steps = 0
        pe = []
        last_idx = np.zeros(W)
        for step in range(T):
            states = builder.get_idc_observations_batch(
                ego, path_indices=[cidx] * W)
            info = env.sim.info_tensor().to_torch().cpu().numpy()
            for w in range(W):
                c = float(info[w, ego[w], 0]) > 0
                o = float(info[w, ego[w], 1]) > 0
                if c:
                    ncol[w] = True
                    col_steps += 1
                if o:
                    noff[w] = True
                    off_steps += 1
            pe.append(np.array([float(np.hypot(s[IX_DP], s[IX_LAT5]))
                                for s in states]))
            last_idx = np.array([float(s[IX_IDX]) for s in states])
            acts = agent.select_action(states, deterministic=True)
            env.step_dynamics(extend_action_to_3d(acts))
        return dict(ncol=int(ncol.sum()), col_steps=col_steps,
                    noff=int(noff.sum()), off_steps=off_steps,
                    comp=float(np.mean(np.clip(last_idx, 0, T)) / T * 100),
                    pe=float(np.nanmedian(np.asarray(pe))))

    print(f"{'配置':28s} {'碰撞世界':>7s} {'碰撞步':>6s} {'压线世界':>7s} "
          f"{'压线步':>6s} {'完成度':>7s} {'pos_err':>8s}")
    # 单配置单进程：避免一个进程里连跑多次 rollout 互相污染
    r = run_once(args.geo, args.speed)
    print(f"{'RESULT':8s} geo={args.geo:7s} speed={args.speed:7s} "
          f"ncol={r['ncol']:3d} col_steps={r['col_steps']:4d} "
          f"noff={r['noff']:3d} off_steps={r['off_steps']:4d} "
          f"comp={r['comp']:5.1f}% pos_err={r['pe']:.3f}")


if __name__ == "__main__":
    main()
