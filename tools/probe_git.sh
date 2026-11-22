#!/bin/bash
echo "===== 1) IDC-Bridge 仓库 ====="
cd /home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge 2>/dev/null && {
  git rev-parse --abbrev-ref HEAD 2>/dev/null
  git remote -v 2>/dev/null | head -2
  echo "--- status ---"
  git status --porcelain 2>/dev/null | head -40
  echo "--- 统计 ---"
  git status --porcelain 2>/dev/null | wc -l
}
echo
echo "===== 2) plant2 仓库 ====="
cd /home/jiangchengxuan/PycharmProjects/research-safe-drive/plant2 2>/dev/null && {
  git rev-parse --abbrev-ref HEAD 2>/dev/null
  git remote -v 2>/dev/null | head -2
  echo "--- status ---"
  git status --porcelain 2>/dev/null | head -40
  echo "--- 统计 ---"
  git status --porcelain 2>/dev/null | wc -l
}
echo
echo "===== 3) 我们新写的文件在哪 ====="
ls -la /home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/tools/*.py 2>/dev/null | grep -E 'plant|route|vector|build_|eval_|plot_|train_plant' | head -20
ls -la /home/jiangchengxuan/PycharmProjects/research-safe-drive/plant2/carla_garage/plant_agent_vec.py 2>/dev/null
echo
echo "===== 4) 数据/产物（不进 git）====="
du -sh /mnt/2T_HDD/plant2_vector/* 2>/dev/null | head -10
