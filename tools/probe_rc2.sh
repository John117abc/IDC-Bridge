#!/bin/bash
cd /gpudrive/baselines/IDC-Bridge || exit 1
echo "=== pdms.py 里 route_completion 上下文 ==="
sed -n 130,180p src/metrics/pdms.py
