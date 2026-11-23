#!/bin/bash
set -u
C=idc_hist
CR=/gpudrive/baselines/IDC-Bridge
echo "=== 所有 config 里的 planner 键 ==="
docker exec $C bash -lc "grep -rn 'planner' $CR/src/configs/*.yaml | head -10"
echo
echo "=== base.yaml 顶层结构（前 40 行）==="
docker exec $C bash -lc "sed -n '1,40p' $CR/src/configs/base.yaml"
echo
echo "=== __main__ 里剩余的 CLI 参数 ==="
docker exec $C bash -lc "n=\$(grep -n '__main__' $CR/src/scripts/eval/idc-eval.py | cut -d: -f1); sed -n \"\$((n+30)),\$((n+70))p\" $CR/src/scripts/eval/idc-eval.py"
