#!/bin/bash
cd /gpudrive/baselines/IDC-Bridge || exit 1
grep -n "_select_gif_worlds" src/scripts/eval/idc-eval.py
echo ---
sed -n "$(grep -n 'def _select_gif_worlds' src/scripts/eval/idc-eval.py | cut -d: -f1),+16p" src/scripts/eval/idc-eval.py
