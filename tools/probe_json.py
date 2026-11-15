"""探查 Waymo JSON 的字段结构，看有没有车道关联信息（Q1 是否可救）。"""
import glob
import json
import sys


def brief(v, depth=0):
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}: {brief(x, depth+1)}"
                               for k, x in list(v.items())[:12]) + "}"
    if isinstance(v, list):
        if len(v) == 0:
            return "[]"
        return f"[{len(v)} x {brief(v[0], depth+1)}]"
    if isinstance(v, (int, float, str, bool)) or v is None:
        return f"{type(v).__name__}"
    return type(v).__name__


def main():
    files = sorted(glob.glob("/data/data_json/training/tfrecord-*.json"))
    if not files:
        print("没找到 json")
        return
    path = files[0]
    print("file:", path.split("/")[-1])
    d = json.load(open(path))
    print("top keys:")
    for k, v in d.items():
        n = len(v) if hasattr(v, "__len__") else "-"
        print(f"  {k:28s} {type(v).__name__:8s} len={n}")
    print()
    for k in ("roadgraph_samples", "lane", "lanes", "map", "metadata"):
        if k in d:
            print(f"[{k}] -> {brief(d[k])[:400]}")
            print()
    # roadgraph 里若有 id / type，看类型分布
    rg = d.get("roadgraph_samples")
    if isinstance(rg, dict):
        for kk, vv in rg.items():
            if isinstance(vv, list):
                uniq = len(set(vv[:5000]))
                print(f"  roadgraph.{kk}: len={len(vv)} 前5000个唯一值={uniq}")
    # 轨迹字段
    for k in ("state", "object_vector", "track", "objects"):
        if k in d:
            print(f"[{k}] -> {brief(d[k])[:300]}")


if __name__ == "__main__":
    main()
