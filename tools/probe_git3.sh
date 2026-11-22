#!/bin/bash
A=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge
B=/home/jiangchengxuan/PycharmProjects/research-safe-drive/plant2
for p in "$A" "$B"; do
  echo "================ $p"
  echo "分支: $(git -C "$p" rev-parse --abbrev-ref HEAD)  上游: $(git -C "$p" rev-parse --abbrev-ref --symbolic-full-name @{u} 2>/dev/null || echo none)"
  git -C "$p" status --porcelain | head -60
  echo "--- 变更行数（已跟踪文件）---"
  git -C "$p" diff --stat | tail -5
done
