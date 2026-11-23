#!/bin/bash
set -u
cd /home/jiangchengxuan/PycharmProjects/research-safe-drive/plant2
echo "=== carla_garage 相关文件 ==="
ls -la carla_garage/ | grep -E "pid_exec|idc|frames|plant_agent" | head -10
echo "=== 是否被 gitignore ==="
git check-ignore -v carla_garage/carla_pid_exec.py 2>/dev/null || echo "未被忽略"
echo "=== .gitignore 改动 ==="
git diff .gitignore | head -20
