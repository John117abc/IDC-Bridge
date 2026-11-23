#!/bin/bash
set -u
C=idc_hist
LOG=/workspace/data/logs/train_84_long.log
echo "=== 进程 ==="
docker exec $C bash -lc "pgrep -af 'idc-train.py' | grep -v pgrep | head -2 || echo 已结束"
echo "=== 日志规模 / 完成轮数 ==="
docker exec $C bash -lc "wc -l $LOG 2>/dev/null; grep -c '步数 91/91' $LOG 2>/dev/null"
echo "=== PDMS 全曲线（每轮）==="
docker exec $C bash -lc "grep -oE 'PDMS\] score=[0-9.]+' $LOG | tail -35"
echo "=== 末尾 ==="
docker exec $C bash -lc "tail -6 $LOG"
echo "=== 新保存的权重 ==="
docker exec $C bash -lc "ls -lat /workspace/data/checkpoints/ | head -4; find /workspace/data/checkpoints -newermt '-10 hours' -name '*.pth' 2>/dev/null | tail -5"
