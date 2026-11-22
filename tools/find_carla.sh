#!/bin/bash
echo "===== 全盘找 CarlaUE4.sh（深度 5） ====="
find / -maxdepth 5 -name 'CarlaUE4.sh' 2>/dev/null | head -10
echo "===== 找 carla 0.9.15 的 python 包（egg/whl） ====="
find / -maxdepth 7 \( -name 'carla-0.9.15*' -o -name 'carla-0.9.16*' \) 2>/dev/null | head -12
echo "===== 找 carla 源码包目录 ====="
find / -maxdepth 6 -type d -name 'carla' -path '*PythonAPI*' 2>/dev/null | head -8
echo "===== 各环境的 python 版本 ====="
for e in /mnt/2T_HDD/conda_envs/*/; do
  v=$("${e}bin/python" --version 2>&1 | head -1)
  [ -n "$v" ] && echo "$(basename "$e") -> $v"
done
echo "===== bench2drive_b2d 里 carla 装在哪 ====="
/mnt/2T_HDD/conda_envs/bench2drive_b2d/bin/python - <<'EOF' 2>&1 | head -6
try:
    import carla, os
    print('version', carla.__version__)
    print('path', os.path.dirname(carla.__file__))
except Exception as e:
    print('fail', e)
EOF
