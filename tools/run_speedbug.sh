#!/bin/bash
set -u
docker exec idc_hist bash -lc 'cd /gpudrive/baselines/IDC-Bridge && timeout 600 /gpudrive/.venv/bin/python tools/diag_speed_bug.py > /tmp/speedbug.log 2>&1; echo exit=$?'
echo "=== 结果 ==="
docker exec idc_hist bash -lc "grep -vE 'Logger|pkg_resources|UserWarning|warnings.warn' /tmp/speedbug.log | tail -26"
