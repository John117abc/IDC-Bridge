#!/bin/bash
set -u
PY=/mnt/2T_HDD/conda_envs/bench2drive_b2d/bin/python
T=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/tools
D=/mnt/2T_HDD/plant2_vector
echo "=== 本次运行的 dbg 文件 ==="
ls -la $D/dbg_*.npz 2>/dev/null
for f in $D/dbg_*.npz; do
  [ -f "$f" ] || continue
  n=$(basename "$f" .npz)
  echo "--- $n ---"
  $PY $T/plot_closedloop.py "$f" "run_$n" 2>&1 | grep -E 'frames|speed mean' | head -3
  $PY $T/plot_plan_ctrl.py 2>&1 | tail -1
  cp $D/carla_plan_ctrl.png $D/plan_$n.png 2>/dev/null
done
echo "=== 产物 ==="
ls -la $D/run_*_view.gif $D/plan_*.png 2>/dev/null | head -10
