#!/usr/bin/env bash
set -u
cd /gpudrive/baselines/IDC-Bridge || exit 1
P=/gpudrive/.venv/bin/python
D=/workspace/data/head60/medoid
L=/workspace/data/logs

$P tools/train_head.py --data "$D" --epochs 200 --bs 128 --lr 2e-4 \
    --out "$D/head.pt" > "$L/train_medoid.log" 2>&1
echo "train exit=$?" >> "$L/medoid_eval.log"
tail -1 "$L/train_medoid.log" >> "$L/medoid_eval.log"

$P tools/eval_head_pdms.py --data "$D" --model "$D/head.pt" \
    --data-dir /data/data_json/training --n-scenes 150 --horizon 40 --no-resid \
    > "$L/eval_medoid_h40.log" 2>&1
echo "eval h40 exit=$?" >> "$L/medoid_eval.log"
tail -8 "$L/eval_medoid_h40.log" >> "$L/medoid_eval.log"
