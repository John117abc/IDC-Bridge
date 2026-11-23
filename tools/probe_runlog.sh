#!/bin/bash
set -u
C=idc_hist
echo "=== idc-train.log 末尾 ==="
docker exec $C bash -lc "ls -lat /workspace/data/logs/20261122/ | head -6"
docker exec $C bash -lc "tail -25 /workspace/data/logs/20261122/idc-train.log"
echo "=== 步数/回合计数 ==="
docker exec $C bash -lc "grep -c '步数' /workspace/data/logs/20261122/idc-train.log || true"
