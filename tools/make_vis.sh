#!/bin/bash
set -u
T=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/tools
PY=/mnt/2T_HDD/conda_envs/bench2drive_b2d/bin/python
D=/mnt/2T_HDD/plant2_vector
# 用帧数最多的那份 dump（1773，跑得久）
F=$(ls -S $D/dbg_*.npz 2>/dev/null | head -1)
echo "使用 dump: $F"
$PY $T/plot_closedloop.py "$F" carla_idc 2>&1 | tail -4
ls -la $D/carla_idc_view.gif $D/carla_idc_frames.png 2>/dev/null
