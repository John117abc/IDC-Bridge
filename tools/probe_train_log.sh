#!/bin/bash
set -u
C=idc_hist
echo "=== 日志行数 ==="
docker exec $C bash -lc "wc -l /tmp/train_v2.log"
echo "=== 末尾 30 行 ==="
docker exec $C bash -lc "tail -30 /tmp/train_v2.log"
echo "=== 步数/回合痕迹 ==="
docker exec $C bash -lc "grep -cE '步数' /tmp/train_v2.log; grep -E '回合|步数' /tmp/train_v2.log | tail -4"
echo "=== 保存的模型 ==="
docker exec $C bash -lc "ls -lat /workspace/data/checkpoints/ | head -5"
