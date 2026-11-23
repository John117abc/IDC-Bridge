#!/bin/bash
R=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge
echo "=== 找 state builder ==="
grep -rIl --include=*.py -e "class IDCStateBuilder" -e "state_mode" -e "window_size" "$R/src" 2>/dev/null | head -5
echo "=== 找 62 维/state 维度定义 ==="
grep -rIn --include=*.py -e "62" -e "DIM_" -e "def get_state" -e "def build_state" "$R/src/env" 2>/dev/null | head -20
echo "=== 候选路径/参考相关 ==="
grep -rIn --include=*.py -e "def _candidate" -e "def install_external" -e "curvature" "$R/src/env/candidate_sampler.py" 2>/dev/null | head -12
ls "$R/src/env" | head -20
