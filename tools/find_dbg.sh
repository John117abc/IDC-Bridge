#!/bin/bash
set -u
D=$(ls -dt /mnt/2T_HDD/plant2_vector/closedloop/idcroute_* 2>/dev/null | head -1)
echo "最新 run 目录: $D"
ls -la "$D" | head -5
echo "=== IDCdbg 行 ==="
grep -h 'IDCdbg' "$D"/*.log 2>/dev/null | head -8
echo "..."
grep -h 'IDCdbg' "$D"/*.log 2>/dev/null | tail -4
echo "=== 计数 ==="
grep -hc 'IDCdbg' "$D"/*.log 2>/dev/null | head -3
