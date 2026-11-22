#!/bin/bash
# 找 IDC checkpoint / world_files.json / 上次外部候选产物
ls /workspace/data/ | head -40
echo "--- pth ---"
find /workspace/data -maxdepth 3 -type f -name '*.pth' | head -8
echo "--- world_files ---"
find /workspace -maxdepth 3 -type f -name 'world_files*.json' 2>/dev/null | head -5
echo "--- 上次外部候选 npz ---"
ls -la /workspace/data/head_plans2.npz /workspace/data/head_plans*.npz 2>/dev/null | head -5
