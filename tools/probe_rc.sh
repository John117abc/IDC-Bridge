#!/bin/bash
cd /gpudrive/baselines/IDC-Bridge || exit 1
echo "=== RC / route completion 相关 ==="
grep -rn "route_completion\|Route Completion\|off_route\|offroad" src/scripts/eval/idc-eval.py | head -12
echo "=== 谁算的 ==="
grep -rn "def .*route\|route_completion" src/env/*.py src/metrics/*.py 2>/dev/null | head -12
