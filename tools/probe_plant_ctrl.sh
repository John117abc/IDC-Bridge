#!/bin/bash
set -u
R=/home/jiangchengxuan/PycharmProjects/research-safe-drive/plant2
echo "=== PlanT agent 里的控制器 ==="
grep -nE "class .*Controller|class PID|def _get_steer|def _get_throttle|def get_control|turn_controller|pid" $R/carla_garage/plant_agent.py 2>/dev/null | head -20
echo
echo "=== 控制器实现（toolkit / utils）==="
grep -rnE "class PIDController|class .*Controller" $R/carla_garage/*.py 2>/dev/null | head -10
echo
echo "=== 关键常量（aim_distance / lookahead / 速度上限）==="
grep -nE "aim_distance|lookahead|max_speed|target_speed|K_P|K_D|K_I" $R/carla_garage/plant_agent.py 2>/dev/null | head -20
