#!/bin/bash
set -u
PY=/mnt/2T_HDD/conda_envs/bench2drive_b2d/bin/python
T=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/tools
D=/mnt/2T_HDD/plant2_vector/vector_ds_full
L=/mnt/2T_HDD/plant2_vector
echo "启动训练 $(date)"
nohup "$PY" "$T/train_plant2_small.py" --data "$D" --epochs 3 --bs 64 \
  > "$L/train_full.log" 2>&1 &
sleep 30
tail -5 "$L/train_full.log"
