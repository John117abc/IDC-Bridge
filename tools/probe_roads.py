"""探查 roads 元素结构：是折线还是孤立线段？（决定 Q1 能否救）"""
import glob
import json
from collections import Counter


def main():
    path = sorted(glob.glob("/data/data_json/training/tfrecord-*.json"))[0]
    d = json.load(open(path))
    roads = d["roads"]
    print(f"file={path.split('/')[-1]}  roads={len(roads)}")
    r0 = roads[0]
    print("\nroads[0] keys:", list(r0.keys()))
    for k, v in r0.items():
        n = len(v) if hasattr(v, "__len__") else "-"
        t = type(v).__name__
        print(f"  {k:16s} {t:8s} len={n}")
        if isinstance(v, list) and v:
            print(f"      first item: {str(v[0])[:120]}")
    # 类型分布 + 折线长度分布
    types = Counter()
    lens = []
    for r in roads:
        types[r.get("type", r.get("road_type", "?"))] += 1
        pts = r.get("geometry") or r.get("points") or r.get("polyline")
        if isinstance(pts, list):
            lens.append(len(pts))
    print("\ntype 分布:", dict(types))
    if lens:
        lens.sort()
        print(f"折线点数: n={len(lens)} min={lens[0]} p50={lens[len(lens)//2]} "
              f"max={lens[-1]}")
    # 是否含 connectivity / lane id
    for key in ("connectivity", "id", "lane_id", "entry_lanes", "exit_lanes"):
        have = sum(1 for r in roads if key in r)
        if have:
            print(f"含字段 {key}: {have}/{len(roads)}")


if __name__ == "__main__":
    main()
