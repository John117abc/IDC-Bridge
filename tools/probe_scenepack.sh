#!/bin/bash
set -u
echo "=== ScenePack 里 seg_a 的来源 ==="
grep -nE "seg_a|roadgraph|map|seg_lane" /home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/src/env/planner_features.py | head -20

echo
echo "=== 样本 json 的顶层键 ==="
docker exec idc_hist bash -lc '
/gpudrive/.venv/bin/python - <<PY
import json
p = "/data/data_json/training/tfrecord-00000-of-01000_100.json"
d = json.load(open(p))
print("top keys:", list(d.keys())[:15])
for k in d:
    v = d[k]
    if isinstance(v, list):
        print(f"  {k}: list len={len(v)}", (type(v[0]).__name__ if v else ""))
    elif isinstance(v, dict):
        print(f"  {k}: dict keys={list(v.keys())[:8]}")
PY'
