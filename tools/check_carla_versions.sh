#!/bin/bash
echo "===== /mnt/2T_HDD 下的 CARLA 目录 ====="
ls -d /mnt/2T_HDD/CARLA* 2>/dev/null
for d in /mnt/2T_HDD/CARLA*; do
  echo "--- $d"
  ls "$d" | head -6
  cat "$d/VERSION" 2>/dev/null
done
echo "===== 各版本 PythonAPI dist（egg/wheel） ====="
for d in /mnt/2T_HDD/CARLA*; do
  echo "--- $d/PythonAPI/carla/dist"
  ls "$d/PythonAPI/carla/dist" 2>/dev/null | head -8
done
echo "===== 各 conda 环境里的 carla 版本 ====="
for e in /mnt/2T_HDD/conda_envs/*/; do
  v=$("${e}bin/python" -c "import carla,sys;print(carla.__version__)" 2>/dev/null)
  [ -n "$v" ] && echo "$(basename $e) -> $v"
done
echo "===== 正在运行的 CARLA 版本 ====="
pgrep -af CarlaUE4 | head -2
