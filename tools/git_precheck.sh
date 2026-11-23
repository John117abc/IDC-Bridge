#!/bin/bash
set -u
cd /home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge
echo "=== tools/src 里的未跟踪大文件（>1MB）==="
git status --porcelain --untracked-files=all tools src | awk '{print $2}' | while read -r f; do
  [ -f "$f" ] && sz=$(stat -c %s "$f") && [ "$sz" -gt 1048576 ] && echo "$sz  $f"
done | sort -rn | head -10
echo "=== src 变更 ==="
git status --porcelain src | head -20
echo "=== plant2 carla_garage 变更 ==="
cd /home/jiangchengxuan/PycharmProjects/research-safe-drive/plant2
git status --porcelain carla_garage | head -20
echo "=== plant2 大文件检查 ==="
git status --porcelain --untracked-files=all carla_garage | awk '{print $2}' | tr -d '"' | while read -r f; do
  [ -f "$f" ] && sz=$(stat -c %s "$f") && [ "$sz" -gt 1048576 ] && echo "$sz  $f"
done | sort -rn | head -5
