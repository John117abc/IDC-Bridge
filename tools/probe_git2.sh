#!/bin/bash
for p in /home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge \
         /home/jiangchengxuan/PycharmProjects/research-safe-drive/plant2 \
         /home/jiangchengxuan/PycharmProjects/research-safe-drive \
         /home/jiangchengxuan/PycharmProjects/gpudrive; do
  echo "== $p"
  git -C "$p" rev-parse --show-toplevel 2>&1 | head -1
  git -C "$p" remote -v 2>/dev/null | head -1
done
echo
echo "== plant2 是否有 .git =="
ls -d /home/jiangchengxuan/PycharmProjects/research-safe-drive/plant2/.git 2>/dev/null || echo "no .git in plant2"
ls -d /home/jiangchengxuan/PycharmProjects/research-safe-drive/.git 2>/dev/null || echo "no .git in research-safe-drive"
