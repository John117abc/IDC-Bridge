#!/usr/bin/env bash
# 今晚自动流水线：等训练完 → 训一版带 checkpoint → 开环 PDMS 评测 → 扩数据重训 → 再评测
set -u
cd /gpudrive/baselines/IDC-Bridge || exit 1
P=/gpudrive/.venv/bin/python
OUT=/workspace/data
L=$OUT/logs
mkdir -p "$L"

echo "===== [NIGHT] start $(date '+%F %T') =====" >> "$L/night.log"

# 0) 等当前训练结束
for i in $(seq 1 90); do
    if ! pgrep -f train_head.py >/dev/null; then break; fi
    sleep 20
done
echo "[NIGHT] 训练已结束 $(date '+%T')" >> "$L/night.log"

# 1) 训一版并保存 checkpoint（20k 数据）
$P tools/train_head.py --data "$OUT/head20" --epochs 200 --bs 128 --lr 2e-4 \
    --out "$OUT/head20/head.pt" > "$L/train_save.log" 2>&1
echo "[NIGHT] train_save exit=$? $(date '+%T')" >> "$L/night.log"
tail -1 "$L/train_save.log" >> "$L/night.log"

# 2) 开环 PDMS 评测（head vs oracle vs const_vel vs expert）
$P tools/eval_head_pdms.py --data "$OUT/head20" --model "$OUT/head20/head.pt" \
    --data-dir /data/data_json/training --n-scenes 300 \
    > "$L/eval_head_20k.log" 2>&1
echo "[NIGHT] eval_20k exit=$? $(date '+%T')" >> "$L/night.log"
tail -8 "$L/eval_head_20k.log" >> "$L/night.log"

# 3) 扩数据到 60k（I/O 密集，约 40~60 分钟）
$P tools/build_head_dataset.py --data-dir /data/data_json/training \
    --n-scenes 60000 --out "$OUT/head60" --k 16 > "$L/build60.log" 2>&1
echo "[NIGHT] build60 exit=$? $(date '+%T')" >> "$L/night.log"
tail -3 "$L/build60.log" >> "$L/night.log"

# 4) 60k 重训
$P tools/train_head.py --data "$OUT/head60" --epochs 300 --bs 128 --lr 2e-4 \
    --out "$OUT/head60/head.pt" > "$L/train60.log" 2>&1
echo "[NIGHT] train60 exit=$? $(date '+%T')" >> "$L/night.log"
tail -1 "$L/train60.log" >> "$L/night.log"

# 5) 60k 版评测
$P tools/eval_head_pdms.py --data "$OUT/head60" --model "$OUT/head60/head.pt" \
    --data-dir /data/data_json/training --n-scenes 300 \
    > "$L/eval_head_60k.log" 2>&1
echo "[NIGHT] eval_60k exit=$? $(date '+%T')" >> "$L/night.log"
tail -8 "$L/eval_head_60k.log" >> "$L/night.log"

echo "===== [NIGHT] DONE $(date '+%F %T') =====" >> "$L/night.log"
