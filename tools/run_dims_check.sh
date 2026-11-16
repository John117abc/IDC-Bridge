#!/usr/bin/env bash
export MADRONA_MWGPU_KERNEL_CACHE=/gpudrive/gpudrive_cache
cd /gpudrive/baselines/IDC-Bridge || exit 1
P=/gpudrive/.venv/bin/python
$P tools/check_openloop_pdms.py --data-dir /data/data_json/training \
    --n-scenes 60 --seed 0 > /workspace/data/logs/ol_dims.log 2>&1
echo "openloop exit=$?"
$P tools/diag_zero_cause.py --data-dir /data/data_json/training \
    --n-scenes 24 --seed 0 > /workspace/data/logs/zc_dims.log 2>&1
echo "zerocause exit=$?"
