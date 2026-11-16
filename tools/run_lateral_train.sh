#!/usr/bin/env bash
# 横向多模态候选训练（warmup + ε 衰减），从 baseline 权重热启动，崩了自动续训。
set -u
export MADRONA_MWGPU_KERNEL_CACHE=/gpudrive/gpudrive_cache
PY=/gpudrive/.venv/bin/python
TRAIN=/gpudrive/baselines/IDC-Bridge/src/scripts/train/idc-train.py
OUT=/workspace/data
LOG=$OUT/logs/lateral_train.log

TOTAL=3000                       # 2400(baseline) + 600(横向)
BASE=$OUT/checkpoints/20261113/idc-waymo-v1.0_examples_220948_episode=2400.pth
TODAY=$(date +%Y%m%d)
CKPT_DIR=$OUT/checkpoints/$TODAY
mkdir -p "$CKPT_DIR"

for attempt in $(seq 1 20); do
    CKPT=$(ls -1 "$CKPT_DIR"/*.pth 2>/dev/null \
        | sed -E 's/.*episode=([0-9]+)\.pth/\1\t&/' | sort -n | tail -1 | cut -f2-)
    if [ -z "${CKPT:-}" ] && [ -f "$BASE" ]; then
        CKPT="$BASE"
    fi
    echo "===== [SUP] attempt=$attempt $(date '+%F %T') ckpt=${CKPT:-none} =====" >> "$LOG"
    "$PY" "$TRAIN" --data-dir /data/data_json/training --file-dir "$OUT" \
        --device cuda --epochs "$TOTAL" --max-bad-worlds 1000 \
        --resample-interval 10 --resample-ratio 0.4 --save-freq 20 \
        --load-model --model-path "$CKPT" >> "$LOG" 2>&1
    rc=$?
    echo "===== [SUP] attempt=$attempt exit=$rc $(date '+%F %T') =====" >> "$LOG"
    if [ "$rc" -eq 0 ]; then break; fi
    sleep 15
done
