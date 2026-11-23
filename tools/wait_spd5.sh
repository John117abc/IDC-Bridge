#!/bin/bash
set -u
C=idc_hist
LOG=/workspace/data/logs/train_84_spd5.log
sleep 660
echo "=== 训练进度 ==="
docker exec $C bash -lc "grep -c '步数 91/91' $LOG; grep -oE 'PDMS\] score=[0-9.]+' $LOG | tail -6"
echo "=== 进程 ==="
docker exec $C bash -lc "pgrep -af idc-train.py | grep -v pgrep | head -1 >/dev/null && echo 仍在运行 || echo 已完成"
echo "=== 最新权重 ==="
docker exec $C bash -lc "ls -t /workspace/data/checkpoints/20261123/*.pth 2>/dev/null | head -2; ls -t /workspace/data/checkpoints/20261122/*.pth 2>/dev/null | head -1"
