#!/bin/bash
set -u
docker exec idc_hist bash -lc "grep -A20 'in load' /tmp/eval_v2.log | head -26"
