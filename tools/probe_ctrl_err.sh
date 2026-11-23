#!/bin/bash
set -u
C=idc_hist
echo "=== v1 对照的报错 ==="
docker exec $C bash -lc "grep -A10 'Traceback' /tmp/ctrl_v1.log | head -14"
echo "=== base 对照的报错 ==="
docker exec $C bash -lc "grep -A10 'Traceback' /tmp/ctrl_base.log | head -14; tail -3 /tmp/ctrl_base.log"
