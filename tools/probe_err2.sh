#!/bin/bash
set -u
D=$(ls -dt /mnt/2T_HDD/plant2_vector/closedloop/gain_1.0_* 2>/dev/null | head -1)
echo "run: $D"
grep -B2 -A12 'Traceback' "$D"/run.log 2>/dev/null | head -24
