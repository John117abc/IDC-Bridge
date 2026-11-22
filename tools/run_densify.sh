#!/bin/bash
PY=/mnt/2T_HDD/conda_envs/bench2drive_b2d/bin/python
T=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/tools
N=/mnt/2T_HDD/plant2_vector/vector_ds/multi_NonSignalizedJunctionRightTurn.npz
"$PY" "$T/plot_densify_compare.py" --npz "$N" --idx 40 \
  --out /mnt/2T_HDD/plant2_vector/densify_compare.png 2>&1 | tail -4
ls -la /mnt/2T_HDD/plant2_vector/densify_compare.png
