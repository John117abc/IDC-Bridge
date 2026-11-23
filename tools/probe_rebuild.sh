#!/bin/bash
set -u
CR=/gpudrive/baselines/IDC-Bridge
echo "=== install_external 是否重建 ref_tensor ==="
docker exec idc_hist bash -lc "grep -n 'rebuild_paths_tensor\\|def install_external' $CR/src/env/candidate_sampler.py | head -8"
echo "=== 训练脚本里 install 候选的时机 ==="
docker exec idc_hist bash -lc "grep -n 'rebuild_paths_tensor\\|_maybe_install_candidates' $CR/src/scripts/train/idc-train.py | head -8"
echo "=== 训练日志里的 ref 诊断 ==="
docker exec idc_hist bash -lc "grep -E 'DIAG-ref|CANDS' /tmp/train_v2b.log | head -8"
