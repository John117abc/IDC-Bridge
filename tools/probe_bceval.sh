#!/bin/bash
set -u
D=$(ls -dt /mnt/2T_HDD/plant2_vector/closedloop/bc_eval_* 2>/dev/null | head -1)
echo "run: $D"
grep -B2 -A10 'Traceback' "$D"/run.log 2>/dev/null | head -20
echo "--- 关键词 ---"
grep -E 'Error|error' "$D"/run.log 2>/dev/null | head -5
