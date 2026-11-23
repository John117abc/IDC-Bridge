#!/bin/bash
set -u
R=/home/jiangchengxuan/PycharmProjects/research-safe-drive/plant2/carla_garage
echo "=== LateralPIDController ==="
sed -n "$(grep -n 'class LateralPIDController' $R/lateral_controller.py | cut -d: -f1),+45p" $R/lateral_controller.py
echo
echo "=== LongitudinalPIDController ==="
sed -n "$(grep -n 'class LongitudinalPIDController' $R/longitudinal_controller.py | cut -d: -f1),+55p" $R/longitudinal_controller.py
