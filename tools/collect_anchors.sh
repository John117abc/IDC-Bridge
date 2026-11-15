#!/usr/bin/env bash
# 按 seed 分进程采集专家轨迹（每进程只能建一个 env）
set -u
PY=/gpudrive/.venv/bin/python
SCRIPT=/gpudrive/baselines/IDC-Bridge/tools/collect_expert_traj.py
OUT=/workspace/data/anchors
N=${1:-8}
W=${2:-120}

mkdir -p "$OUT"
for s in $(seq 0 $((N - 1))); do
    "$PY" "$SCRIPT" --data-dir /data/data_json/training \
        --seed "$s" --num-worlds "$W" --out-dir "$OUT" 2>&1 \
        | grep -E 'collect|Error|Traceback' || true
done
echo "[collect] done"
ls -l "$OUT"
