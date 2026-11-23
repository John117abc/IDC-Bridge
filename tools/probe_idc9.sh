#!/bin/bash
R=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge
F=$R/src/agents/idc_agent.py
echo "=== _encode ==="
sed -n "$(grep -n 'def _encode' $F | cut -d: -f1),+40p" $F
echo
echo "=== _physical_action ==="
sed -n "$(grep -n 'def _physical_action' $F | cut -d: -f1),+30p" $F
echo
echo "=== _norm / input_norm 定义 ==="
grep -nE "class .*Norm|def _norm|input_norm *=" $F | head -8
sed -n "$(grep -n 'def _norm' $F | head -1 | cut -d: -f1),+12p" $F
