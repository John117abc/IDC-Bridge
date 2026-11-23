#!/bin/bash
R=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge
echo "=== A) rebuild_paths_tensor 里 curv_ahead/spd_ahead/road_ahead 的定义 ==="
grep -nE "curv_ahead|spd_ahead|road_ahead" "$R/src/env/idc_state_builder.py" | head -20
echo
echo "--- 上下文（719-800）---"
sed -n '719,800p' "$R/src/env/idc_state_builder.py"

echo
echo "=== B) get_road_dist_batch（1171-1195）==="
sed -n '1171,1196p' "$R/src/env/idc_state_builder.py"

echo
echo "=== C) idc_agent 动作解码（_policy 之后，300-360）==="
sed -n '300,360p' "$R/src/agents/idc_agent.py"
