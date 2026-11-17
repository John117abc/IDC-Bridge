#!/usr/bin/env bash
set -u
cd /gpudrive/baselines/IDC-Bridge || exit 1
P=/gpudrive/.venv/bin/python
D=/workspace/data/head_dd/medoid
L=/workspace/data/logs

echo "===== [MEDOID-DD] start $(date '+%F %T') =====" >> "$L/medoid_dd.log"
$P tools/train_head.py --data "$D" --epochs 150 --bs 128 --lr 2e-4 --patience 8 \
    --out "$D/head_best.pt" > "$L/train_medoid_dd.log" 2>&1
echo "[MEDOID-DD] train exit=$? $(date '+%T')" >> "$L/medoid_dd.log"
tail -3 "$L/train_medoid_dd.log" >> "$L/medoid_dd.log"

$P tools/eval_head_pdms.py --data "$D" --model "$D/head_best.pt" \
    --data-dir /data/data_json/training --n-scenes 150 --horizon 40 --no-resid \
    > "$L/eval_medoid_dd.log" 2>&1
echo "[MEDOID-DD] eval exit=$? $(date '+%T')" >> "$L/medoid_dd.log"
tail -9 "$L/eval_medoid_dd.log" >> "$L/medoid_dd.log"
echo "===== [MEDOID-DD] DONE $(date '+%F %T') =====" >> "$L/medoid_dd.log"
