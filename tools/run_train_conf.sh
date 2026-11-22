#!/bin/bash
set -u
PY=/mnt/2T_HDD/conda_envs/bench2drive_b2d/bin/python
T=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/tools
L=/mnt/2T_HDD/plant2_vector
pkill -f train_plant2_full 2>/dev/null
sleep 1
echo "启动置信度头训练 $(date)"
nohup "$PY" -u "$T/train_plant2_conf.py" --epochs 40 --bs 64 \
  > "$L/train40_conf.log" 2>&1 &
sleep 60
tail -6 "$L/train40_conf.log"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
