#!/bin/bash
set -u
PY=/mnt/2T_HDD/conda_envs/bench2drive_b2d/bin/python
T=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/tools
cd /home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge || exit 1
$PY "$T/train_plant2_small.py" --epochs 3 2>&1 | tail -12
