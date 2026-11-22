#!/bin/bash
echo "===== /mnt/2T_HDD 顶层 ====="
ls -la /mnt/2T_HDD 2>/dev/null | head -20
echo "===== 各目录大小 ====="
du -sh /mnt/2T_HDD/* 2>/dev/null | sort -h | tail -15
echo "===== 找 carla_garage 风格数据（boxes/measurements 目录） ====="
find /mnt/2T_HDD -maxdepth 6 -type d -name 'boxes' 2>/dev/null | head -5
find /mnt/2T_HDD -maxdepth 6 -type d -name 'measurements' 2>/dev/null | head -5
echo "===== 找 .xodr 地图 ====="
find /mnt/2T_HDD -maxdepth 6 -name '*.xodr' 2>/dev/null | head -8
echo "===== 找 route xml / Bench2Drive 路线 ====="
find /mnt/2T_HDD -maxdepth 6 -name '*.xml' -path '*route*' 2>/dev/null | head -5
find /mnt/2T_HDD -maxdepth 6 -name 'results.json.gz' 2>/dev/null | head -3
echo "===== PlanT2 数据集候选目录 ====="
find /mnt/2T_HDD -maxdepth 4 -type d \( -iname '*plant*' -o -iname '*carla*' -o -iname '*dataset*' \) 2>/dev/null | head -12
