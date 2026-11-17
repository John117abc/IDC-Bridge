#!/usr/bin/env bash
set -u
cd /gpudrive/baselines/IDC-Bridge || exit 1
P=/gpudrive/.venv/bin/python
OUT=/workspace/data/head_dd
L=/workspace/data/logs

echo "===== [DD] start $(date '+%F %T') =====" >> "$L/dd_align.log"

# 1) 重建数据：8 waypoint(0.5s) + K=20
$P tools/build_head_dataset.py --data-dir /data/data_json/training \
    --n-scenes 60000 --out "$OUT" --k 20 > "$L/build_dd.log" 2>&1
echo "[DD] build exit=$? $(date '+%T')" >> "$L/dd_align.log"
tail -5 "$L/build_dd.log" >> "$L/dd_align.log"

# 2) 训练
$P tools/train_head.py --data "$OUT" --epochs 300 --bs 128 --lr 2e-4 \
    --out "$OUT/head.pt" > "$L/train_dd.log" 2>&1
echo "[DD] train exit=$? $(date '+%T')" >> "$L/dd_align.log"
tail -2 "$L/train_dd.log" >> "$L/dd_align.log"

# 3) 评测（4 秒时域）
$P tools/eval_head_pdms.py --data "$OUT" --model "$OUT/head.pt" \
    --data-dir /data/data_json/training --n-scenes 150 --horizon 40 --no-resid \
    > "$L/eval_dd.log" 2>&1
echo "[DD] eval exit=$? $(date '+%T')" >> "$L/dd_align.log"
tail -9 "$L/eval_dd.log" >> "$L/dd_align.log"

echo "===== [DD] DONE $(date '+%F %T') =====" >> "$L/dd_align.log"
