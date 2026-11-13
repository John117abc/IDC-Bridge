#!/usr/bin/env bash
# 无人值守训练看护脚本：崩了就从最新 checkpoint 自动续训。
# 用法：bash tools/overnight.sh
set -u

PY=/gpudrive/.venv/bin/python
TRAIN=/gpudrive/baselines/IDC-Bridge/src/scripts/train/idc-train.py
DATA=/data/data_json/training
OUT=/workspace/data
LOG=$OUT/logs/overnight.log
TODAY=$(date +%Y%m%d)
CKPT_DIR=$OUT/checkpoints/$TODAY

TOTAL=2400          # 总轮数
INTERVAL=10         # 每 10 轮滚动替换一次场景
RATIO=0.5           # 每次替换一半 world
SAVE=20             # 每 20 轮存一次权重

mkdir -p "$CKPT_DIR"

for attempt in $(seq 1 30); do
    # 按 episode 编号取最大（文件名里 episode=N 不一定在末尾，不能用 sort -V）。
    CKPT=$(ls -1 "$CKPT_DIR"/*.pth 2>/dev/null \
        | sed -E 's/.*episode=([0-9]+)\.pth/\1\t&/' \
        | sort -n | tail -1 | cut -f2-)
    echo "===== [SUPERVISOR] attempt=$attempt $(date '+%F %T') ckpt=${CKPT:-none} =====" >> "$LOG"
    if [ -z "${CKPT:-}" ]; then
        "$PY" "$TRAIN" --data-dir "$DATA" --file-dir "$OUT" --device cuda \
            --epochs "$TOTAL" --resample-interval "$INTERVAL" --resample-ratio "$RATIO" \
            --max-bad-worlds 1000 --save-freq "$SAVE" >> "$LOG" 2>&1
    else
        "$PY" "$TRAIN" --data-dir "$DATA" --file-dir "$OUT" --device cuda \
            --epochs "$TOTAL" --resample-interval "$INTERVAL" --resample-ratio "$RATIO" \
            --max-bad-worlds 1000 --save-freq "$SAVE" \
            --load-model --model-path "$CKPT" >> "$LOG" 2>&1
    fi
    rc=$?
    echo "===== [SUPERVISOR] attempt=$attempt exit=$rc $(date '+%F %T') =====" >> "$LOG"
    if [ "$rc" -eq 0 ]; then
        echo "[SUPERVISOR] 正常结束" >> "$LOG"
        break
    fi
    sleep 15
done
