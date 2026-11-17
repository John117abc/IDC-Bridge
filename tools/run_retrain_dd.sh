#!/usr/bin/env bash
set -u
cd /gpudrive/baselines/IDC-Bridge || exit 1
P=/gpudrive/.venv/bin/python
D=/workspace/data/head_dd
L=/workspace/data/logs

echo "===== [RETRAIN] start $(date '+%F %T') =====" >> "$L/retrain_dd.log"
$P tools/train_head.py --data "$D" --epochs 150 --bs 128 --lr 2e-4 --patience 8 \
    --out "$D/head_best.pt" > "$L/train_dd2.log" 2>&1
echo "[RETRAIN] train exit=$? $(date '+%T')" >> "$L/retrain_dd.log"
tail -3 "$L/train_dd2.log" >> "$L/retrain_dd.log"

$P tools/eval_head_pdms.py --data "$D" --model "$D/head_best.pt" \
    --data-dir /data/data_json/training --n-scenes 150 --horizon 40 --no-resid \
    > "$L/eval_dd2.log" 2>&1
echo "[RETRAIN] eval exit=$? $(date '+%T')" >> "$L/retrain_dd.log"
tail -9 "$L/eval_dd2.log" >> "$L/retrain_dd.log"
echo "===== [RETRAIN] DONE $(date '+%F %T') =====" >> "$L/retrain_dd.log"
