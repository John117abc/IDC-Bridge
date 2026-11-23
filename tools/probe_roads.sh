#!/bin/bash
set -u
docker exec idc_hist bash -lc '
/gpudrive/.venv/bin/python - <<PY
import json, collections
p = "/data/data_json/training/tfrecord-00000-of-01000_100.json"
d = json.load(open(p))
rs = d["roads"]
print("roads 数:", len(rs))
print("第一段键:", list(rs[0].keys()))
print("示例:", {k: (v if not isinstance(v, list) else f"list[{len(v)}]") for k, v in rs[0].items()})
c = collections.Counter(r.get("type") for r in rs)
print("type 分布:", dict(c))
print("有 geometry 的比例:", sum(1 for r in rs if r.get("geometry")) / len(rs))
PY'
