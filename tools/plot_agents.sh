#!/bin/bash
PY=/mnt/2T_HDD/conda_envs/bench2drive_b2d/bin/python
T=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/tools
D=/mnt/2T_HDD/plant2_vector
for i in 0 1 2 3; do
  echo "== dbg_$i (with agents)"
  "$PY" "$T/plot_closedloop.py" "$D/dbg_$i.npz" "ag$i" 2>&1 | tail -2
done
ls -la "$D"/ag*_frames.png "$D"/ag*_view.gif 2>/dev/null | head -8
