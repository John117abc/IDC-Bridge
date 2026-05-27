"""离线密度扫描：遍历 Waymo tfrecord 文件，统计每个场景的周车密度并输出 JSON。

用法:
    # 全量扫描
    python scan_density.py --data-dir /path/to/tfrecords --output density.json --batch-size 150

    # 随机抽样
    python scan_density.py --data-dir /path/to/tfrecords --output density.json --sample 2000 --batch-size 150
"""

import sys
import os
import json
import glob
import random
import time
import argparse

sys.path.insert(0, '/workspace/idc/src')
from env.env_utils import get_env_config
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.dataset import SceneDataLoader

import torch
import numpy as np


def scan_density(data_dir, output_path, batch_size, sample_size=None, seed=42):
    all_files = sorted(glob.glob(os.path.join(data_dir, "tfrecord*")))
    if sample_size:
        all_files = random.Random(seed).sample(all_files, min(sample_size, len(all_files)))
    total = len(all_files)
    print(f"扫描 {total} 个文件...")

    data_loader = SceneDataLoader(
        root=data_dir,
        batch_size=batch_size,
        dataset_size=total,
        sample_with_replacement=False,
        shuffle=False,
        seed=seed,
    )
    env_config = get_env_config()
    env = GPUDriveTorchEnv(
        config=env_config,
        data_loader=data_loader,
        max_cont_agents=1,
        device="cuda",
        action_type="continuous",
    )

    density = {}
    first_batch = list(env.data_batch)
    start_time = time.time()
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch = all_files[batch_start:batch_end]
        if len(batch) < batch_size:
            batch = batch + [all_files[0]] * (batch_size - len(batch))
        env.swap_data_batch(data_batch=batch)
        env.reset()
        p_np = env.sim.partner_observations_tensor().to_torch().cpu().numpy()
        for w in range(batch_size):
            fidx = batch_start + w
            if fidx < total:
                p = p_np[w, 0]
                valid = (p[:, 0] != 0.0) | (p[:, 1] != 0.0) | (p[:, 2] != 0.0)
                density[all_files[fidx]] = int(valid.sum())
        torch.cuda.empty_cache()
        done = batch_end
        elapsed = time.time() - start_time
        pct = done / total * 100
        eta = (elapsed / done) * (total - done) if done > 0 else 0
        print(f"  [{done}/{total}] {pct:.1f}%  耗时: {elapsed:.0f}s  预计剩余: {eta:.0f}s", flush=True)

    env.swap_data_batch(data_batch=first_batch)
    env.reset()
    torch.cuda.empty_cache()

    values = list(density.values())
    values.sort()
    n = len(values)

    buckets = {"0": 0, "1-2": 0, "3-5": 0, "6-10": 0, "11-20": 0, "21-30": 0, "31+": 0}
    for v in values:
        if v == 0:           buckets["0"] += 1
        elif v <= 2:         buckets["1-2"] += 1
        elif v <= 5:         buckets["3-5"] += 1
        elif v <= 10:        buckets["6-10"] += 1
        elif v <= 20:        buckets["11-20"] += 1
        elif v <= 30:        buckets["21-30"] += 1
        else:                buckets["31+"] += 1

    summary = {
        "total": n,
        "mean": round(sum(values) / n, 2) if n else 0,
        "median": values[n // 2] if n else 0,
        "min": values[0] if n else 0,
        "max": values[-1] if n else 0,
        "buckets": buckets,
    }

    output = {"summary": summary, "files": density}
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"已保存: {output_path}")
    print(f"  文件数: {n}")
    print(f"  均值: {summary['mean']}, 中位数: {summary['median']}")
    print(f"  范围: {summary['min']} - {summary['max']}")
    print(f"  分布: {buckets}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Waymo 场景周车密度离线扫描")
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--output', type=str, required=True, help="输出 JSON 路径")
    parser.add_argument('--batch-size', type=int, default=150)
    parser.add_argument('--sample', type=int, default=None, help="随机抽样数量（不指定=全量）")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    scan_density(args.data_dir, args.output, args.batch_size, args.sample, args.seed)
