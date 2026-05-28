import os
import json
import random

import numpy as np
import torch


class WorldManager:
    def __init__(self, env, builder, agent, all_files, args, logger, compute_density=True):
        self.env = env
        self.builder = builder
        self.agent = agent
        self.all_files = all_files
        self.args = args
        self.logger = logger

        self.num_worlds = args.num_worlds
        self.bad_worlds = set()
        self.ego_indices = None

        self.filter_threshold = getattr(args, 'filter_threshold', 200)

        if compute_density:
            self.density_cache = self._load_or_compute_density_cache()
            self.dense_files = self._build_dense_pool()
        else:
            self.density_cache = {}
            self.dense_files = []

    def _load_or_compute_density_cache(self):
        density_file = getattr(self.args, 'density_cache_file', None)
        if density_file and os.path.exists(density_file):
            with open(density_file, 'r') as f:
                data = json.load(f)
            cache = data.get('files', data)
            self.logger.info(f'密度缓存已加载: {len(cache)} 个世界 (from {density_file})')
            return cache

        cache_file = os.path.join(self.args.file_dir, "world_density.json")
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                data = json.load(f)
            cache = data.get('files', data)
            self.logger.info(f'密度缓存已加载: {len(cache)} 个世界')
            return cache

        probe_size = min(2000, len(self.all_files))
        probe_files = random.sample(self.all_files, probe_size)
        self.logger.info(
            f'正在计算世界密度缓存（随机抽样 {probe_size}/{len(self.all_files)} 个文件，约 30 秒）...'
        )

        cache = {}
        first_batch = list(self.env.data_batch)
        batch_size = self.num_worlds
        for batch_start in range(0, probe_size, batch_size):
            batch_end = min(batch_start + batch_size, probe_size)
            batch = probe_files[batch_start:batch_end]
            if len(batch) < batch_size:
                batch = batch + [probe_files[0]] * (batch_size - len(batch))
            self.env.swap_data_batch(data_batch=batch)
            self.env.reset()
            p_np = self.env.sim.partner_observations_tensor().to_torch().cpu().numpy()
            for w in range(batch_size):
                fidx = batch_start + w
                if fidx < probe_size:
                    p = p_np[w, 0]
                    valid = (p[:, 0] != 0.0) | (p[:, 1] != 0.0) | (p[:, 2] != 0.0)
                    cache[probe_files[fidx]] = float(valid.sum())
            torch.cuda.empty_cache()

        self.env.swap_data_batch(data_batch=first_batch)
        self.env.reset()
        torch.cuda.empty_cache()

        with open(cache_file, 'w') as f:
            json.dump({"files": cache}, f)
        self.logger.info(f'密度缓存已保存: {len(cache)} 个世界')
        return cache

    def _build_dense_pool(self):
        min_d = self.args.min_partner_density
        max_d = getattr(self.args, 'max_partner_density', 30)
        size = self.args.dense_sample_size

        if max_d == 30 and min_d == 0 and size >= len(self.density_cache):
            files = sorted(self.density_cache.keys(),
                          key=lambda f: self.density_cache.get(f, 0), reverse=True)
        else:
            files = sorted(
                [f for f, d in self.density_cache.items()
                 if min_d <= d <= max_d],
                key=lambda f: self.density_cache[f], reverse=True
            )
            if 0 < size < len(files):
                files = files[:size]

        self.logger.info(
            f'世界候选池: {len(files)} 个 (density ∈ [{min_d}, {max_d}], cap={size if size > 0 else "无"})'
        )
        return files

    def filter_initial(self, ego_indices):
        self.ego_indices = ego_indices
        self.bad_worlds.clear()
        self.logger.info('Pre-training world filter: checking path coordinates...')

        for w in range(self.num_worlds):
            a = ego_indices[w]
            for pid in range(self.builder.num_candidate_paths):
                pos = self.builder.candidate_paths[w][a][pid]['pos']
                if np.max(np.abs(pos[:, 0])) > self.filter_threshold or np.max(np.abs(pos[:, 1])) > self.filter_threshold:
                    self.bad_worlds.add(w)
                    self.logger.warning(
                        f'[FILTER-path] world_{w} path_{pid} '
                        f'max|pos|=({np.max(np.abs(pos[:,0])):.0f},{np.max(np.abs(pos[:,1])):.0f})'
                    )
                    break

        self.logger.info(f'[FILTER] {len(self.bad_worlds)}/{self.num_worlds} worlds excluded (path anomaly)')

    def filter_per_step(self, states, step):
        ref_start = self.agent.DIM_EGO + self.agent.DIM_OTHERS + self.agent.DIM_VALIDITY

        for w in range(self.num_worlds):
            if w in self.bad_worlds:
                continue
            if abs(states[w][0]) > self.filter_threshold or abs(states[w][1]) > self.filter_threshold:
                self.bad_worlds.add(w)
                dp = abs(states[w][ref_start])
                self.logger.debug(
                    f'[FILTER-ego] world_{w} step={step} '
                    f'ego=({states[w][0]:.0f},{states[w][1]:.0f}) delta_p={dp:.0f}'
                )

        if step % 5 == 0:
            max_dp, max_w = 0.0, 0
            for w in range(self.num_worlds):
                if w in self.bad_worlds:
                    continue
                s = states[w]
                dp = abs(s[ref_start])
                if dp > max_dp:
                    max_dp, max_w = dp, w
            self.logger.info(
                f'[DIAG-ref] max pos_err={max_dp:.2f}m @world_{max_w} '
                f'good_worlds={self.good_count}/{self.num_worlds}'
            )

    def should_resample(self):
        return len(self.bad_worlds) > self.args.max_bad_worlds

    def resample(self):
        self.logger.info(
            f'[RESAMPLE] Triggered: {len(self.bad_worlds)}/{self.num_worlds} bad worlds. Reloading...'
        )

        for attr in ['ref_tensor', 'expert_pos', 'expert_vel', 'expert_heading',
                      'candidate_paths', '_road_cache']:
            if hasattr(self.builder, attr):
                delattr(self.builder, attr)
        torch.cuda.empty_cache()

        candidate_pool = (
            self.dense_files
            if self.agent.rho > 0 and len(self.dense_files) >= self.num_worlds * 2
            else self.all_files
        )
        batch_files = random.sample(candidate_pool, self.num_worlds)
        self.env.swap_data_batch(data_batch=batch_files)

        new_ego = []
        for w in range(self.num_worlds):
            ctrl = self.env.cont_agent_mask[w].nonzero(as_tuple=True)[0]
            new_ego.append(ctrl[0].item() if len(ctrl) > 0 else 0)

        self.builder._setup_expert_data()
        self.builder.clear_cache()
        self.builder.generate_candidate_paths(new_ego, num_paths=3)

        self.bad_worlds.clear()
        for w in range(self.num_worlds):
            a = new_ego[w]
            for pid in range(self.builder.num_candidate_paths):
                pos = self.builder.candidate_paths[w][a][pid]['pos']
                if np.max(np.abs(pos[:, 0])) > self.filter_threshold or np.max(np.abs(pos[:, 1])) > self.filter_threshold:
                    self.bad_worlds.add(w)
                    break

        self.env.reset()
        for w in range(self.num_worlds):
            self.builder.reset_world_step(w, 0)
        self.builder.clear_cache()

        self.agent.update_ego_indices(new_ego)
        self.agent.clear_buffer()

        self.logger.info(
            f'[RESAMPLE] complete: {len(self.bad_worlds)} bad, '
            f'{self.good_count} good'
        )
        self.ego_indices = new_ego
        return new_ego

    @property
    def good_worlds(self):
        return [w for w in range(self.num_worlds) if w not in self.bad_worlds]

    @property
    def good_count(self):
        return self.num_worlds - len(self.bad_worlds)
