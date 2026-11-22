#!/bin/bash
cd /gpudrive/baselines/IDC-Bridge || exit 1
echo "=== files mentioning cand0 ==="
grep -rln 'cand0' . 2>/dev/null | head -10
echo "=== storyboard: cand / path drawing ==="
grep -n 'cand\|top-1\|candidate' src/utils/storyboard.py 2>/dev/null | head -20
echo "=== storyboard: trace keys ==="
grep -n 'trace\|frames\|gif' src/utils/storyboard.py 2>/dev/null | head -20
