#!/bin/bash
set -u
D=$(ls -dt /mnt/2T_HDD/plant2_vector/closedloop/idcroute_* 2>/dev/null | head -1)
echo "run: $D"
echo "=== 实际应用的 HUD（agent 记录的 thr/brk/v）==="
grep -hoE 'step [0-9]+ v=[0-9.]+ mode=[0-9]+ logit=[-0-9.]+ steer=[-0-9.]+ thr=[0-9.]+ brk=[0-9.]+' "$D"/*.log 2>/dev/null | head -6
echo "..."
grep -hoE 'step [0-9]+ v=[0-9.]+ mode=[0-9]+ logit=[-0-9.]+ steer=[-0-9.]+ thr=[0-9.]+ brk=[0-9.]+' "$D"/*.log 2>/dev/null | tail -4
echo "=== 车速统计 ==="
grep -hoE 'v=[0-9.]+' "$D"/*.log 2>/dev/null | sed 's/v=//' | awk '{s+=$1;n++; if($1>m)m=$1} END{if(n)printf "均值 %.2f 最大 %.2f (n=%d)\n", s/n, m, n}'
