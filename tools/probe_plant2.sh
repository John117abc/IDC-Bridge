#!/bin/bash
R=/home/jiangchengxuan/PycharmProjects/research-safe-drive/plant2
echo "===== 顶层 ====="
ls -la "$R" | head -30
echo "===== 一层目录 ====="
find "$R" -maxdepth 2 -type d -not -path '*/.git*' -not -path '*/__pycache__*' | head -40
echo "===== README ====="
[ -f "$R/README.md" ] && head -50 "$R/README.md"
echo "===== 关键文件 ====="
find "$R" -maxdepth 3 -name '*.yaml' -o -maxdepth 3 -name '*.yml' -o -maxdepth 3 -name 'setup.py' -o -maxdepth 3 -name 'requirements*.txt' -o -maxdepth 3 -name 'pyproject.toml' | head -20
echo "===== 是否有 carla / 控制器 / 仿真 ====="
grep -rIl --include=*.py -e 'carla' -e 'Carla' "$R" 2>/dev/null | head -10
echo "--- controller/planner 类名 ---"
grep -rIn --include=*.py -e 'class .*Controller' -e 'class .*Planner' -e 'class .*Agent' "$R" 2>/dev/null | head -20
echo "===== 数据/日志目录 ====="
du -sh "$R"/* 2>/dev/null | sort -h | tail -12
