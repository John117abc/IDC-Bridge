#!/bin/bash
set -u
PY=/mnt/2T_HDD/conda_envs/bench2drive_b2d/bin/python
T=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/tools
O=/mnt/2T_HDD/plant2_vector/vector_ds_full
mkdir -p "$O"
echo "=== ?????? $(date) ==="
$PY "$T/build_plant2_dataset.py" --scenarios all --max-routes 40 --out "$O" 2>&1 | grep -vE 'first_seg_angle'
echo "=== ???? $(date) ==="
ls "$O" | wc -l
du -sh "$O"
