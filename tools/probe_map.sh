#!/bin/bash
set -u
echo "=== 1) eval 里的步进调用 ==="
grep -nE "step_dynamics|env\.step\(|\.step\(" /home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/src/scripts/eval/idc-eval.py | head -8

echo
echo "=== 2) EnvConfig 里地图相关字段 ==="
grep -nE "roadgraph|map_|polygon|top_k|feature" /gpudrive/gpudrive/env/env_config.py 2>/dev/null | head -15 || \
docker exec idc_hist bash -lc 'grep -rnE "roadgraph|map_" /gpudrive/gpudrive/env/env_config.py | head -15'

echo
echo "=== 3) planner_features 里怎么取地图 ==="
grep -nE "map_observation|map_|roadgraph" /home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/src/env/planner_features.py | head -12
