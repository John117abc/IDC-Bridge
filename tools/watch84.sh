#!/bin/bash
set -u
C=idc_hist
LOG=/workspace/data/logs/train_84_long.log
sleep 240
echo "=== 进度 ==="
docker exec $C bash -lc "grep -cE '步数 91/91' $LOG; grep -E 'PDMS\]' $LOG | tail -8"
echo "=== 最近保存 ==="
docker exec $C bash -lc "grep -E '保存模型' $LOG | tail -3"
echo "=== 错误（若有）==="
docker exec $C bash -lc "grep -A6 Traceback $LOG | head -10"
