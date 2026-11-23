#!/bin/bash
set -u
echo "=== IDC-Bridge 状态 ==="
cd /home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge
git status --porcelain | head -40
echo
echo "=== plant2 状态 ==="
cd /home/jiangchengxuan/PycharmProjects/research-safe-drive/plant2
git status --porcelain | head -30
echo
echo "=== 两个仓库的 remote ==="
git -C /home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge remote -v | head -2
git -C /home/jiangchengxuan/PycharmProjects/research-safe-drive/plant2 remote -v | head -2
