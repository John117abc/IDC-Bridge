#!/bin/bash
set -u
T=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/tools
PY=/mnt/2T_HDD/conda_envs/bench2drive_b2d/bin/python
"$PY" "$T/build_plant2_dataset.py" \
  --max-routes 3 \
  --out /mnt/2T_HDD/plant2_vector/vector_ds 2>&1 | tail -20
ls -la /mnt/2T_HDD/plant2_vector/vector_ds/*.npz | tail -6
