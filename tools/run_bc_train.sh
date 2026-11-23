#!/bin/bash
set -u
R=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge
PY=/mnt/2T_HDD/conda_envs/bench2drive_b2d/bin/python
CK=$(ls -t /mnt/2T_HDD/WaymoDatabas/checkpoints/20261123/*.pth 2>/dev/null | head -1)
echo "基础权重 = $CK"
$PY $R/tools/bc_finetune.py \
  --data /mnt/2T_HDD/plant2_vector/bc \
  --ckpt "$CK" \
  --out /mnt/2T_HDD/WaymoDatabas/checkpoints/bc/idc_bc.pth \
  --epochs 40 --batch 256 --lr 1e-3 2>&1 | tail -35
