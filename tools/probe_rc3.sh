#!/bin/bash
cd /gpudrive/baselines/IDC-Bridge || exit 1
echo "=== last_temporal 赋值处 ==="
grep -rn "last_temporal\|temporal" src/metrics/pdms.py | head -12
echo "=== eval 里传进来的 temporal ==="
grep -rn "temporal\|update(" src/scripts/eval/idc-eval.py | head -14
