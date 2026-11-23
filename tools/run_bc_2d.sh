#!/bin/bash
set -u
R=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge
PY=/mnt/2T_HDD/conda_envs/bench2drive_b2d/bin/python
$PY - <<'EOF'
import ast
ast.parse(open('/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/tools/bc_2d.py').read())
print('AST OK')
EOF
$PY $R/tools/bc_2d.py \
  --data /mnt/2T_HDD/plant2_vector/bc \
  --ckpt /mnt/2T_HDD/WaymoDatabas/checkpoints/bc/idc_bc.pth \
  --out /mnt/2T_HDD/WaymoDatabas/checkpoints/bc/idc_2d.pth \
  --epochs 30 --batch 256 --lr 8e-4 2>&1 | tail -14
