#!/bin/bash
set -u
L=/workspace/data/logs/train_84_v2.log
echo "完成轮数: $(docker exec idc_hist bash -lc "grep -c '步数 91/91' $L")"
echo "最近 6 轮 PDMS:"
docker exec idc_hist bash -lc "grep -oE 'PDMS\] score=[0-9.]+' $L | tail -6"
echo "进程:"
docker exec idc_hist bash -lc "pgrep -af idc-train.py | grep -v pgrep | head -1 || echo 已结束"
