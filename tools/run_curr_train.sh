#!/usr/bin/env bash
# 从头训 + 课程（无任何专家未来参考）：
#   阶段1 0~599    : K=1（中心地图候选）
#   阶段2 600~1199 : K=5 + ε 探索（eps 从 600 起算）
set -u
export MADRONA_MWGPU_KERNEL_CACHE=/gpudrive/gpudrive_cache
PY=/gpudrive/.venv/bin/python
TRAIN=/gpudrive/baselines/IDC-Bridge/src/scripts/train/idc-train.py
OUT=/workspace/data
LOG=$OUT/logs/curr_train2.log

TOTAL=1200
TODAY=$(date +%Y%m%d)
CKPT_DIR=$OUT/checkpoints/${TODAY}_curr2
mkdir -p "$CKPT_DIR"

for attempt in $(seq 1 20); do
    CKPT=$(ls -1 "$CKPT_DIR"/*.pth 2>/dev/null \
        | sed -E 's/.*episode=([0-9]+)\.pth/\1\t&/' | sort -n | tail -1 | cut -f2-)
    if [ -n "${CKPT:-}" ]; then
        RESUME="--load-model --model-path $CKPT"
    else
        RESUME=""
    fi
    echo "===== [SUP] attempt=$attempt $(date '+%F %T') ckpt=${CKPT:-scratch} =====" >> "$LOG"
    "$PY" "$TRAIN" --data-dir /data/data_json/training --file-dir "$OUT" \
        --device cuda --epochs "$TOTAL" --max-bad-worlds 1000 \
        --resample-interval 10 --resample-ratio 0.4 --save-freq 20 \
        $RESUME >> "$LOG" 2>&1
    rc=$?
    echo "===== [SUP] attempt=$attempt exit=$rc $(date '+%F %T') =====" >> "$LOG"
    if [ "$rc" -eq 0 ]; then break; fi
    sleep 15
done
