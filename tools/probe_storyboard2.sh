#!/bin/bash
cd /gpudrive/baselines/IDC-Bridge || exit 1
echo "=== idc-eval: render_world / candidates / trace ==="
grep -n 'render_world\|candidates\|trace\[' src/scripts/eval/idc-eval.py | head -25
echo "=== storyboard render_world 签名与调用处 ==="
sed -n '210,240p' src/utils/storyboard.py
