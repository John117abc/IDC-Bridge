#!/bin/bash
set -u
docker exec idc_hist bash -lc 'cd /gpudrive/baselines/IDC-Bridge && timeout 600 /gpudrive/.venv/bin/python tools/diag_pack.py > /tmp/diag_pack.log 2>&1; echo exit=$?'
echo "=== 关键输出 ==="
docker exec idc_hist bash -lc "grep -vE 'Logger|pkg_resources|UserWarning|warnings.warn' /tmp/diag_pack.log | tail -18"
