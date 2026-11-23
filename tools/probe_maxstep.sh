#!/bin/bash
set -u
CR=/gpudrive/baselines/IDC-Bridge
echo "=== max_step 定义 ==="
docker exec idc_hist bash -lc "grep -n 'max_step' $CR/src/scripts/train/idc-train.py | head -10"
echo "=== epochs / current_epoch 定义 ==="
docker exec idc_hist bash -lc "grep -nE 'epochs =|current_epoch =' $CR/src/scripts/train/idc-train.py | head -10"
