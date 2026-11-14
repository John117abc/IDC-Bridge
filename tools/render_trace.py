"""从 eval 落盘的 trace npz 离线渲染 storyboard / GIF。

trace 由 ``idc-eval.py`` 在 ``<file_dir>/traces/YYYYMMDD/trace_*.npz`` 生成，
内含每个 world 的逐帧 ego/伙伴位姿、事件标记、地图点云与专家路径。

用法：
    python tools/render_trace.py traces/trace_151354.npz -o plots/story
    python tools/render_trace.py <npz> -o <dir> --worlds 3,5 --no-gif
"""
import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import numpy as np  # noqa: E402

from utils.storyboard import render_world  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", help="trace npz 路径")
    ap.add_argument("-o", "--out", default="plots/story", help="输出目录")
    ap.add_argument("--worlds", default="", help="只渲染指定 world，逗号分隔")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--margin", type=float, default=35.0,
                    help="视野半径（米）")
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--no-storyboard", action="store_true")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    worlds = sorted(int(k.split("_")[1]) for k in d.files if k.startswith("ego_"))
    if args.worlds:
        want = {int(x) for x in args.worlds.split(",") if x.strip()}
        worlds = [w for w in worlds if w in want]
    if not worlds:
        raise SystemExit(f"{args.npz} 里没有可渲染的 world")

    for w in worlds:
        tw = {"ego": d[f"ego_{w}"], "info": d[f"info_{w}"],
              "partners": d[f"partners_{w}"]}
        png, gif = render_world(tw, d[f"map_{w}"], d[f"expert_{w}"],
                                args.out, w, fps=args.fps, margin=args.margin,
                                storyboard=not args.no_storyboard,
                                gif=not args.no_gif)
        if png:
            print("saved", png)
        if gif:
            print("saved", gif)


if __name__ == "__main__":
    main()
