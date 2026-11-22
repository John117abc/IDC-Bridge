#!/bin/bash
PY=/mnt/2T_HDD/conda_envs/bench2drive_b2d/bin/python
T=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/tools
"$PY" "$T/plot_vector_sample.py" \
  --npz /mnt/2T_HDD/plant2_vector/vector_ds/smoke_Town12.npz \
  --idx 40 --out /mnt/2T_HDD/plant2_vector/sample_view.png 2>&1 | tail -3
ls -la /mnt/2T_HDD/plant2_vector/sample_view.png
