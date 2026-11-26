#!/bin/bash
set -u
D=/mnt/2T_HDD/plant2_vector
RD=$(ls -dt $D/closedloop/fast_* 2>/dev/null | head -1)
echo "run: $RD"
grep -E "score_composed|duration_game" "$RD/run.json" 2>/dev/null | head -4
echo "=== 平均速度（PID 行）==="
grep -oE "v=[0-9.]+" "$RD/run.log" 2>/dev/null | awk -F= '{s+=$2;n++} END{printf "%.2f m/s (%d 样本)\n", s/n, n}'
echo "=== 结果表 ==="
grep -E "RouteCompletionTest|CollisionTest|MinSpeedTest" "$RD/run.log" | tail -3
