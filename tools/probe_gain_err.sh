#!/bin/bash
set -u
D=$(ls -dt /mnt/2T_HDD/plant2_vector/closedloop/gain_* 2>/dev/null | head -2 | tail -1)
echo "run dir: $D"
grep -A8 'Traceback' "$D"/run.log 2>/dev/null | head -14
echo "=== 或（grep 失败时）最近日志 ==="
grep -A8 'Traceback' /tmp/gain_1.0.log 2>/dev/null | head -14
