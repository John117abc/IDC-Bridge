#!/bin/bash
cd /gpudrive/baselines/IDC-Bridge || exit 1
echo "=== trace ego 来源 ==="
sed -n '505,525p' src/scripts/eval/idc-eval.py
