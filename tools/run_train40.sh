#!/bin/bash
set -u
PY=/mnt/2T_HDD/conda_envs/bench2drive_b2d/bin/python
T=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/tools
L=/mnt/2T_HDD/plant2_vector
pkill -f train_plant2_small 2>/dev/null
sleep 2
echo "启动正式训练 $(date)"
nohup "$PY" -u "$T/train_plant2_full.py" --epochs 40 --bs 64 \
  > "$L/train40.log" 2>&1 &
sleep 45
echo "--- 前 45s 日志 ---"
tail -6 "$L/train40.log"
echo "--- GPU ---"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
