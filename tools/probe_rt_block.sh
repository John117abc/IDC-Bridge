#!/bin/bash
cd /gpudrive/baselines/IDC-Bridge || exit 1
echo "=== _RT 相关代码块 ==="
sed -n "$(grep -n '_RT = None' src/scripts/eval/idc-eval.py | cut -d: -f1),+30p" src/scripts/eval/idc-eval.py
echo "=== step 循环里的 RT 调用 ==="
sed -n "$(grep -n '_RT.observe' src/scripts/eval/idc-eval.py | cut -d: -f1),+10p" src/scripts/eval/idc-eval.py
echo "=== max_step 定义 ==="
grep -n "max_step" src/scripts/eval/idc-eval.py | head -5
