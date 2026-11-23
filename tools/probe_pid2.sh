#!/bin/bash
set -u
R=/home/jiangchengxuan/PycharmProjects/research-safe-drive/plant2/carla_garage
echo "=== lateral_controller.py 方法列表 ==="
grep -nE "class |def " $R/lateral_controller.py | head -12
echo
echo "=== LateralPIDController 主体（取 steer 的公式）==="
sed -n "$(grep -n 'class LateralPIDController' $R/lateral_controller.py | cut -d: -f1),+80p" $R/lateral_controller.py | head -60
echo
echo "=== 相关 config 默认值 ==="
grep -nE "lateral_pid|longitudinal_pid" $R/config.py | head -20
