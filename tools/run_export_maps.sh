#!/bin/bash
# 找能 import carla 的 python，然后导出地图折线到 /mnt/2T_HDD/plant2_vector/maps
OUT=/mnt/2T_HDD/plant2_vector/maps
X1=/mnt/2T_HDD/download/AdditionalMaps_0.9.16/CarlaUE4/Content/Carla/Maps/OpenDrive
X2=/mnt/2T_HDD/CARLA_0.9.16/CarlaUE4/Content/Carla/Maps
TOOLS=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/tools
mkdir -p "$OUT"

echo "===== 找带 carla 的 python ====="
PY=""
for p in /mnt/2T_HDD/conda_envs/*/bin/python /usr/bin/python3 /mnt/2T_HDD/CARLA_0.9.16/PythonAPI/*/python; do
  [ -x "$p" ] || continue
  if "$p" -c "import carla" >/dev/null 2>&1; then echo "FOUND $p"; PY="$p"; break; fi
done
if [ -z "$PY" ]; then
  echo "没找到，尝试把 egg 加进 PYTHONPATH"
  export PYTHONPATH=/mnt/2T_HDD/CARLA_0.9.16/PythonAPI/carla/dist/carla-0.9.16-py3.7-linux-x86_64.egg:$PYTHONPATH
  ls /mnt/2T_HDD/CARLA_0.9.16/PythonAPI/carla/dist/ 2>/dev/null
  for p in /mnt/2T_HDD/conda_envs/*/bin/python /usr/bin/python3; do
    [ -x "$p" ] || continue
    if "$p" -c "import carla" >/dev/null 2>&1; then echo "FOUND(with egg) $p"; PY="$p"; break; fi
  done
fi
[ -z "$PY" ] && { echo "FAIL: 无可用 python+carla"; exit 1; }
"$PY" -c "import carla; print('carla', carla.__version__)"

echo "===== 导出 ====="
"$PY" "$TOOLS/export_map_polylines.py" --xodr-dirs "$X1,$X2" --out "$OUT" 2>&1 | tail -20
ls -la "$OUT" | head -20
