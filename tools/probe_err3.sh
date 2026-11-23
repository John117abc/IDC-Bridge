#!/bin/bash
set -u
D=$(ls -dt /mnt/2T_HDD/plant2_vector/closedloop/gain_1.0_* 2>/dev/null | head -1)
echo "run: $D"
grep -A6 'plant_agent_idc.py", line 50' "$D"/run.log 2>/dev/null | head -10
echo "--- 任何异常信息 ---"
grep -E 'Error|error|Exception|raise' "$D"/run.log 2>/dev/null | head -8
