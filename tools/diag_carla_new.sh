#!/bin/bash
set -u
echo "=== route 模式的 HUD 行（v/thr/brk/steer）==="
grep -oE 'step [0-9]+ v=[0-9.]+ mode=[0-9]+ logit=[-0-9.]+ steer=[-0-9.]+ thr=[0-9.]+ brk=[0-9.]+' /tmp/carla_idc_route_new.log | head -6
echo "..."
grep -oE 'step [0-9]+ v=[0-9.]+ mode=[0-9]+ logit=[-0-9.]+ steer=[-0-9.]+ thr=[0-9.]+ brk=[0-9.]+' /tmp/carla_idc_route_new.log | tail -6
echo
echo "=== 平均车速（HUD 采样）==="
grep -oE 'v=[0-9.]+' /tmp/carla_idc_route_new.log | sed 's/v=//' | awk '{s+=$1;n++} END{if(n)printf "均值 %.2f m/s (n=%d)\n", s/n, n}'
echo "=== IDC 诊断行 ==="
grep -E 'IDC\]' /tmp/carla_idc_route_new.log | head -6
echo "=== 错误 ==="
grep -E 'Traceback|Error' /tmp/carla_idc_route_new.log | head -4
