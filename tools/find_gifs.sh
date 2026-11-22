#!/bin/bash
# 找联合跑产出的 gif/帧
cd /gpudrive/baselines/IDC-Bridge || exit 1
echo "=== gifs 目录（最近）==="
find /workspace/data -maxdepth 3 -name '*.gif' -newermt '-3 hours' 2>/dev/null | head -12
echo "=== 最近 1 小时的所有 gif ==="
find /workspace -maxdepth 4 -name '*.gif' -newermt '-70 minutes' 2>/dev/null | head -12
echo "=== gifs 目录内容 ==="
ls -la --time-style=+%H:%M /workspace/data/gifs 2>/dev/null | tail -8
