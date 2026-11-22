#!/bin/bash
set -u
PY=/mnt/2T_HDD/conda_envs/bench2drive_b2d/bin/python
T=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/tools
L=/mnt/2T_HDD/plant2_vector
"$PY" -u "$T/eval_plant2.py" --limit 3000 --out "$L/eval_report.json" 2>&1 | tail -25
