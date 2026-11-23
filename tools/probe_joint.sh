#!/bin/bash
R=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge
echo "=== planner_runtime.py 接口 ==="
grep -nE "^class |^    def |^def " "$R/src/env/planner_runtime.py" | head -20
echo
echo "=== install_external 签名 ==="
sed -n "$(grep -n 'def install_external' $R/src/env/candidate_sampler.py | cut -d: -f1),+18p" "$R/src/env/candidate_sampler.py"
echo
echo "=== PlannerRuntime.__init__ / plan 关键行 ==="
sed -n "$(grep -n 'class PlannerRuntime' $R/src/env/planner_runtime.py | cut -d: -f1),+40p" "$R/src/env/planner_runtime.py"
