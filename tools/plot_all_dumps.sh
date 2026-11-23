#!/bin/bash
PY=/mnt/2T_HDD/conda_envs/bench2drive_b2d/bin/python
T=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/tools
D=/mnt/2T_HDD/plant2_vector
for i in 0 1 2 3; do
  echo "== dbg_$i"
  "$PY" "$T/plot_closedloop.py" "$D/dbg_$i.npz" "cl$i" 2>&1 | grep -E 'frames|turn frames|speed mean'
done
ls -la "$D"/cl*_frames.png "$D"/cl*_view.gif 2>/dev/null | head -10
