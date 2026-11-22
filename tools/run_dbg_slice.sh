#!/bin/bash
set -u
cd /gpudrive/baselines/IDC-Bridge || exit 1
MODE=route IDS=1 bash tools/run_multi_gif.sh
echo "--- slice 诊断 ---"
grep -E "slice" /workspace/data/logs/multi_route.log | head -8
