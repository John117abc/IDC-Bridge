#!/usr/bin/env bash
set -u
cd /gpudrive/baselines/IDC-Bridge || exit 1
P=/gpudrive/.venv/bin/python
D=/workspace/data/head_dd/medoid64
L=/workspace/data/logs

echo "===== [M64] start $(date '+%F %T') =====" >> "$L/m64.log"
$P tools/train_head.py --data "$D" --epochs 150 --bs 128 --lr 2e-4 --patience 8 \
    --out "$D/head_best.pt" > "$L/train_m64.log" 2>&1
echo "[M64] train exit=$? $(date '+%T')" >> "$L/m64.log"
tail -3 "$L/train_m64.log" >> "$L/m64.log"

$P tools/eval_head_pdms.py --data "$D" --model "$D/head_best.pt" \
    --data-dir /data/data_json/training --n-scenes 150 --horizon 40 --no-resid \
    > "$L/eval_m64.log" 2>&1
echo "[M64] eval exit=$? $(date '+%T')" >> "$L/m64.log"
tail -9 "$L/eval_m64.log" >> "$L/m64.log"
echo "===== [M64] DONE $(date '+%F %T') =====" >> "$L/m64.log"
